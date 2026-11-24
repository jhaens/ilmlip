// ilmlip on the CPU: energy, forces and virial of one structure per call.
//
// The torch model's descriptor and readout by hand, forward and backward, over a Verlet candidate
// list (every pair within rcut + skin, both directions).  One call runs these stages over the
// atoms (OpenMP):
//   PRE    candidates inside rcut, ZBL, layer 0's density and products, the next message
//   MID    layers 1 .. L-2 from the neighbours' messages (L > 2 only)
//   LAST   the last layer's density and products, the features, the readout MLP forward and backward
//   BACK   layers L-2 .. 0: message gradients gathered from the neighbours through reverse bonds
//   FORCE  forces from dE/d(bond) of an atom's bonds and of their reverses; virial
//
// Bounds (model.py): each layer's bond sums are gated by one factor per atom (PRE, MID, LAST; BACK takes the
// factor's derivative, and layer 0's also through the back-flow), the features may be squashed before the
// readout, and the learned energy is switched off by the atom's ZBL energy (FILTER, LAST).
//
// Layer 0 is summed per neighbour species, so a bond costs one outer product.  Nothing is
// scattered: a neighbour gathers what flows to it through the reverse bond.  Bond vectors are
// formed in double precision; forces and virial accumulate in double.
//
// A layer's products work on Blocks (layer 0: one per channel group of a tapered model).  Radial
// weights per ordered pair of species (FiLM, back-flow) are tables made when a structure first has
// the pair, not for all S^2 pairs at load.
//
// Built by ilmlip/cpu.py with g++ -O3 -march=native -fopenmp (GCC >= 8).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

namespace {

// SIMD vectors: u is the same vector at an element's alignment, for loads and stores anywhere.
// GCC 8 copies 32 bytes through the stack in two halves, so each load waits for its store.
template <typename T> struct Simd;
template <> struct Simd<float> {
  typedef float v __attribute__((vector_size(32)));
  typedef float u __attribute__((vector_size(32), aligned(4), may_alias));
  static constexpr int W = 8;
};
template <> struct Simd<double> {
  typedef double v __attribute__((vector_size(32)));
  typedef double u __attribute__((vector_size(32), aligned(8), may_alias));
  static constexpr int W = 4;
};
template <typename T> using vec = typename Simd<T>::v;

template <typename T> inline vec<T> ld(const T* p) {
  return *reinterpret_cast<const typename Simd<T>::u*>(p);
}
template <typename T> inline void st(T* p, vec<T> x) {
  *reinterpret_cast<typename Simd<T>::u*>(p) = x;
}
template <typename T> inline vec<T> splat(T s) { return vec<T>{} + s; }
inline float hsum(vec<float> x) {
  float s = x[0];
  for (int k = 1; k < Simd<float>::W; ++k) s += x[k];
  return s;
}
inline double hsum(vec<double> x) {
  double s = x[0];
  for (int k = 1; k < Simd<double>::W; ++k) s += x[k];
  return s;
}

// the same sums as a tree: fewer dependent additions, other rounding
inline float hsum_tree(vec<float> x) {
#ifdef __AVX__
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
#else
  return hsum(x);
#endif
}
inline double hsum_tree(vec<double> x) {
#ifdef __AVX__
  __m128d s = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
  return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
#else
  return hsum(x);
#endif
}

inline vec<float> vsqrt(vec<float> x) {
#ifdef __AVX__
  return _mm256_sqrt_ps(x);
#else
  for (int k = 0; k < Simd<float>::W; ++k) x[k] = std::sqrt(x[k]);
  return x;
#endif
}
inline vec<double> vsqrt(vec<double> x) {
#ifdef __AVX__
  return _mm256_sqrt_pd(x);
#else
  for (int k = 0; k < Simd<double>::W; ++k) x[k] = std::sqrt(x[k]);
  return x;
#endif
}

inline int64_t round_up(int64_t n, int64_t m) { return (n + m - 1) / m * m; }

// exp, float: Cephes' expf (range reduction by ln 2, degree-6 polynomial), < 2 ulp
inline vec<float> vexp(vec<float> x) {
  typedef int32_t vi __attribute__((vector_size(32)));
  const vec<float> hi = splat(88.3762626647949f), lo = splat(-87.3365447504f);
  x = x < hi ? x : hi;
  x = x > lo ? x : lo;
  const float magic = 12582912.0f;                 // 1.5 * 2^23: rounds to an integer
  vec<float> t = x * 1.44269504088896341f + magic;
  vec<float> n = t - magic;
  vec<float> r = x - n * 0.693359375f;
  r = r - n * -2.12194440e-4f;
  vec<float> p = splat(1.9875691500E-4f);
  p = p * r + 1.3981999507E-3f;
  p = p * r + 8.3334519073E-3f;
  p = p * r + 4.1665795894E-2f;
  p = p * r + 1.6666665459E-1f;
  p = p * r + 5.0000001201E-1f;
  vec<float> y = p * (r * r) + r + 1.0f;
  vi bits = ((vi)t - 0x4B400000 + 127) << 23;      // 2^n
  return y * (vec<float>)bits;
}
inline vec<double> vexp(vec<double> x) {
  for (int k = 0; k < Simd<double>::W; ++k) x[k] = std::exp(x[k]);
  return x;
}

// compile-time vector counts: f(integral_constant<int, n>) for common n, 0 = any n
template <typename F> inline void dispatch(int n, F&& f) {
  switch (n) {
    case 1: f(std::integral_constant<int, 1>()); return;
    case 2: f(std::integral_constant<int, 2>()); return;
    case 3: f(std::integral_constant<int, 3>()); return;
    case 4: f(std::integral_constant<int, 4>()); return;
    case 5: f(std::integral_constant<int, 5>()); return;
    case 6: f(std::integral_constant<int, 6>()); return;
    default: f(std::integral_constant<int, 0>()); return;
  }
}
template <typename F> inline void dispatch_l(int n, F&& f) {
  switch (n) {
    case 2: f(std::integral_constant<int, 2>()); return;
    case 3: f(std::integral_constant<int, 3>()); return;
    default: f(std::integral_constant<int, 0>()); return;
  }
}
template <typename F> inline void dispatch_q(int n, F&& f) {
  switch (n) {
    case 1: f(std::integral_constant<int, 1>()); return;
    case 2: f(std::integral_constant<int, 2>()); return;
    case 3: f(std::integral_constant<int, 3>()); return;
    default: f(std::integral_constant<int, 0>()); return;
  }
}
// f(integral_constant<int, i>) for 0 <= i < N
template <int N, int K = 0, typename F> inline void select(int i, F&& f) {
  if constexpr (K < N) {
    if (i == K)
      f(std::integral_constant<int, K>());
    else
      select<N, K + 1>(i, f);
  }
}
constexpr int isqrt(int n) {
  int r = 0;
  while ((r + 1) * (r + 1) <= n) ++r;
  return r;
}
constexpr int MAXNV = 16;                          // vectors per channel row, any-size path
constexpr int MAXNL = 8;                           // radial l channels, any-size path
constexpr int MAXG = 8;                            // reduction rows per weight group
constexpr int MAXQ = 16;                           // vectors of radial functions or harmonics

// 64-byte aligned buffer
template <typename T> struct Buf {
  T* p = nullptr;
  size_t n = 0;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  Buf& operator=(Buf&& o) noexcept {
    std::swap(p, o.p);
    std::swap(n, o.n);
    return *this;
  }
  ~Buf() { std::free(p); }
  // at least m elements; the contents are not kept when it grows
  void reserve(size_t m) {
    if (m <= n && p) return;
    std::free(p);
    p = nullptr;
    n = 0;
    void* q = nullptr;
    if (posix_memalign(&q, 64, round_up(std::max<size_t>(m, 1) * sizeof(T), 64))) throw std::bad_alloc();
    p = static_cast<T*>(q);
    n = m;
  }
  void zeros(size_t m) {
    reserve(m);
    std::memset(p, 0, m * sizeof(T));
  }
  T* data() { return p; }
  const T* data() const { return p; }
  T& operator[](size_t i) { return p[i]; }
  const T& operator[](size_t i) const { return p[i]; }
};

// the packed model: an int64 stream and a float64 stream
struct Reader {
  const int64_t* I;
  int64_t nI, pi = 0;
  const double* D;
  int64_t nD, pd = 0;
  int64_t i() {
    if (pi >= nI) throw std::runtime_error("model: integer stream too short");
    return I[pi++];
  }
  int ii() { return static_cast<int>(i()); }
  double d() {
    if (pd >= nD) throw std::runtime_error("model: float stream too short");
    return D[pd++];
  }
  void tag(int64_t t) {
    if (i() != t) throw std::runtime_error("model: integer stream out of step at tag " + std::to_string(t));
  }
};

constexpr int64_t MAGIC = 0x67686370;              // "ghcp"
constexpr int64_t VERSION = 5;

// Load-time code, not vectorised: GCC 8's vectoriser hits an internal compiler error here.
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
template <typename N> inline N cst(double v);
template <> inline double cst<double>(double v) { return v; }

constexpr int LMAX_SH = 6;

// The real harmonics of model.angular for the unit vector (X, Y, Z), with any number type N
// that has +, - and * (here: polynomials, below): F_lm P_l^(|m|)(Z) Re/Im (X + iY)^|m|.
template <typename N>
inline void harmonics(int L, N X, N Yc, N Z, N* out) {
  N re[LMAX_SH + 1], im[LMAX_SH + 1];                 // (X + iY)^k
  re[0] = cst<N>(1.0);
  im[0] = cst<N>(0.0);
  for (int k = 1; k <= L; ++k) {
    re[k] = X * re[k - 1] - Yc * im[k - 1];
    im[k] = Yc * re[k - 1] + X * im[k - 1];
  }
  N d[LMAX_SH + 1][LMAX_SH + 1];                      // d[l][k]: k-th derivative of P_l at Z
  double odd = 1.0;                                    // (2k - 1)!!
  for (int k = 0; k <= L; ++k) {
    if (k > 0) odd *= 2.0 * k - 1.0;
    d[k][k] = cst<N>(odd);
    if (k < L) d[k + 1][k] = (2.0 * k + 1.0) * (Z * d[k][k]);
    for (int l = k + 2; l <= L; ++l)
      d[l][k] = (1.0 / (l - k)) * ((2.0 * l - 1.0) * (Z * d[l - 1][k]) - (l + k - 1.0) * d[l - 2][k]);
  }
  int lm = 0;
  for (int l = 0; l <= L; ++l)
    for (int m = -l; m <= l; ++m, ++lm) {
      const int k = m < 0 ? -m : m;
      double f = (2.0 * l + 1.0) * (m == 0 ? 1.0 : 2.0);
      for (int j = l - k + 1; j <= l + k; ++j) f /= j;    // (l - k)! / (l + k)!
      out[lm] = std::sqrt(f) * (d[l][k] * (m < 0 ? im[k] : re[k]));
    }
}

// The harmonics as polynomials in (x, y, z): the recurrences above run once on polynomials
// (dense over the monomials of degree <= LMAX_SH), at load time.
constexpr int NMONO = (LMAX_SH + 1) * (LMAX_SH + 2) * (LMAX_SH + 3) / 6;
struct Monomials {
  int n = 0;                                       // monomials of degree <= L, by degree
  int a[NMONO], b[NMONO], c[NMONO], parent[NMONO], axis[NMONO];
  int index[LMAX_SH + 1][LMAX_SH + 1][LMAX_SH + 1];
  explicit Monomials(int L) {
    for (int d = 0; d <= L; ++d)
      for (int i = d; i >= 0; --i)
        for (int j = d - i; j >= 0; --j) {
          const int k = d - i - j;
          a[n] = i;
          b[n] = j;
          c[n] = k;
          index[i][j][k] = n;
          // one power fewer of the first variable present: x, else y, else z
          if (d == 0) {
            parent[n] = -1;
            axis[n] = -1;
          } else if (i > 0) {
            parent[n] = index[i - 1][j][k];
            axis[n] = 0;
          } else if (j > 0) {
            parent[n] = index[i][j - 1][k];
            axis[n] = 1;
          } else {
            parent[n] = index[i][j][k - 1];
            axis[n] = 2;
          }
          ++n;
        }
  }
};
struct Poly {
  static const Monomials* M;                       // set while the tables are built
  double c[NMONO];
};
const Monomials* Poly::M = nullptr;
inline Poly poly_zero() {
  Poly p;
  std::fill(p.c, p.c + NMONO, 0.0);
  return p;
}
inline Poly operator*(Poly x, Poly y) {
  const Monomials& M = *Poly::M;
  Poly r = poly_zero();
  for (int i = 0; i < M.n; ++i) {
    if (x.c[i] == 0.0) continue;
    for (int j = 0; j < M.n; ++j) {
      if (y.c[j] == 0.0) continue;
      const int A = M.a[i] + M.a[j], B = M.b[i] + M.b[j], C = M.c[i] + M.c[j];
      if (A + B + C > LMAX_SH) throw std::runtime_error("harmonics: degree beyond the table");
      r.c[M.index[A][B][C]] += x.c[i] * y.c[j];
    }
  }
  return r;
}
inline Poly operator*(double s, Poly x) {
  for (double& v : x.c) v *= s;
  return x;
}
inline Poly operator+(Poly x, Poly y) {
  for (int i = 0; i < NMONO; ++i) x.c[i] += y.c[i];
  return x;
}
inline Poly operator-(Poly x, Poly y) {
  for (int i = 0; i < NMONO; ++i) x.c[i] -= y.c[i];
  return x;
}
template <> inline Poly cst<Poly>(double v) {
  Poly p = poly_zero();
  p.c[0] = v;
  return p;
}

#pragma GCC pop_options

// ZBL screened nuclear repulsion of a pair, taken to zero at rmax (model.repulsion)
inline void repulsion(double r, double zi, double zj, double rmax, double* e, double* de) {
  const double KE = 14.399645, A0 = 0.46850;
  const double C[4] = {0.18175, 0.50986, 0.28022, 0.02817};
  const double Dd[4] = {3.19980, 0.94229, 0.40290, 0.20162};
  const double u = r / rmax;
  if (u >= 1.0) {
    *e = *de = 0.0;
    return;
  }
  const double a = A0 / (std::pow(zi, 0.23) + std::pow(zj, 0.23));
  const double x = r / a;
  double phi = 0.0, dphi = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double ex = C[k] * std::exp(-Dd[k] * x);
    phi += ex;
    dphi -= Dd[k] * ex;
  }
  dphi /= a;
  const double w = 1.0 - u;
  const double poly = 1.0 + u * (3.0 + u * (6.0 + u * (10.0 + u * (15.0 + 21.0 * u))));
  const double dpoly = 3.0 + u * (12.0 + u * (30.0 + u * (60.0 + u * 105.0)));
  const double env = w * w * w * poly;
  const double denv = (-3.0 * w * w * poly + w * w * w * dpoly) / rmax;
  const double k = KE * zi * zj;
  *e = k / r * phi * env;
  *de = k * (-phi * env / (r * r) + (dphi * env + phi * denv) / r);
}

template <typename T> struct Engine {
  static constexpr int W = Simd<T>::W;
  static constexpr int MR = 6;                     // GEMM register block: 6 rows x 2 vectors
  static constexpr int NR = 2 * W;
  using V = vec<T>;

  struct Term {                                    // P[o] += cg P[a] P[b]
    int a, b, o;
    T cg;
  };
  struct Use {                                     // row x is an input: gP[x] += cg gP[o] P[other]
    int o, other;
    T cg;
  };
  struct YTerm {                                   // A[o] += cg Y[lm] R[deg lm] I_j[r]
    int lm, r, o, deg;
    T cg;
  };
  // a product basis on C channels, with the radial weights of its density
  struct Block {
    int C = 0, Cp = 0, NV = 0, lo = 0;             // channels, padded, vectors per row, first channel
    int nbase = 0, nrows = 0, nl = 0;              // rows of A, rows of P, radial degrees
    // offsets in a pair table of the radial weights per ordered pair (chem inside):
    // R[l][:] = sum_q g_q tab[pC + (l nb + q) Cp + :], dE/dg += dE/dR[l][c] tab[pQ + (l C + c) nbp + :]
    size_t pC = 0, pQ = 0;
    int lnew = -1;                                 // layer 0: degrees above lnew are in no earlier block
    std::vector<Term> terms;                       // sorted by output row
    std::vector<int> grp;                          // terms of one output row: [grp[g], grp[g + 1])
    std::vector<int> use_off;                      // per row: its uses [use_off[x], use_off[x + 1])
    std::vector<Use> uses;
    std::vector<int> frows;                        // rows of P read out
    int foff = 0;                                  // column of the first feature
    size_t aoff = 0, goff = 0;                     // its A and feature gradients, per-atom rows
  };
  struct Layer {
    std::vector<Block> B;                          // layer 0: one per channel group; later: one
    size_t asz = 0, gsz = 0;                       // per atom: A of every block, feature gradients
    // messages (t >= 1)
    int nI = 0;
    bool bf = false;                               // without the reverse bond's order-1 object
    std::vector<int> reduce_blk, reduce_from, reduce_to;  // reduction rows: block, row -> I[tgt]
    std::vector<size_t> red_off, redT_off;         // where each row's weights start
    Buf<T> redW;                                   // per row [C_{t-1}][Cp]
    Buf<T> redWT;                                  // per row [C][Cp_{t-1}]
    std::vector<int> rg_off, rg_f;                 // reduction rows sharing one weight (by M)
    std::vector<int> rg_blk;                       // groups of block b: [rg_blk[b], rg_blk[b + 1])
    std::vector<YTerm> yt;                         // sorted by output row
    std::vector<YTerm> ytr;                        // with bf: sorted by message row,
    std::vector<int> yr_off;                       // those of row r: [yr_off[r], yr_off[r + 1])
  };
  // the last layer read out by its scalars, Gram matrices and a low-rank bispectrum
  struct Gram {
    int r = 0, rb = 0, rp = 0, npair = 0;
    std::vector<std::vector<int>> rows;            // per degree: rows of A, [function][m]
    std::vector<Buf<T>> G, GT, Bm, BT;             // per degree: [f][c][rp], [f][k][Cp], [f][c][Cp_b], [f][k][Cp]
    std::vector<size_t> poff;                      // per degree: its rows of projections
    size_t psz = 0;
    Block bis;                                     // products of the rank-rb projections
    int foff = 0;
  };
  struct Dense {
    int nin = 0, nout = 0, nout_p = 0, nin_p = 0;
    Buf<T> Wf;                                     // forward panels: [nout_p/NR][nin][NR]
    Buf<T> Wb;                                     // backward panels: [nin_p/NR][nout][NR]
    Buf<T> bias;                                   // [species][nout_p] for layer 0, else [nout_p]
  };

  // model
  int S = 0, L = 0, lmax = 0, lmax_feat = 0, n_lm = 0, n_lm_feat = 0, nlmp = 0, nb = 0, nbp = 0, cheb_p = 5;
  double rcut = 0, scale = 1, zbl_factor = 0;
  bool zbl = false;
  // bounds: B1 gate of every layer's bond sum by one factor per atom, B2 squash of the whitened features,
  // B3 core switch of the learned energy by the atom's ZBL energy
  bool gate = false, squash = false, core = false;
  double tau = 0, zc = 0;
  std::vector<double> gate_c, gate_s2, gate_n;     // per layer: a0 / sigma, (sigma kappa)^2, size of the bond sum
  double gate_k2 = 0;                              // (a0 kappa)^2
  std::vector<T> fmean, fstd;                      // the features' mean and spread, in the engine's order
  std::vector<double> e0, zbl_z, zbl_rcov;
  std::vector<double> vc_scale, vc_rc2;
  std::vector<int> deg;
  // harmonics as polynomials: monomial recipe, then per lm (and per lm and axis) its terms
  int n_mono = 0;
  std::vector<int> mono_parent, mono_axis;
  std::vector<int> sh_off, sh_idx, dsh_off, dsh_idx;
  std::vector<double> sh_c, dsh_c;
  std::vector<Layer> lay;
  Gram gram;
  std::vector<Dense> mlp;
  Buf<T> head;                                     // [species][H_p]
  int F = 0, F_ld = 0;                             // features, row stride of x
  int Hlast = 0;
  // radial weights per ordered pair (key centre * S + neighbour), built when a structure first
  // has the pair: F_l = (W_l (1 + gamma) + W^beta_l beta) chem0[neighbour]; layer 0's blocks and
  // the message layers take channels of F_l, the back-flow F_l M_l
  int nbf = 0;                                     // degrees of the back-flow
  std::vector<int> chan, chan_off;                 // layer 0's channels per degree, offsets in a film row
  std::vector<std::vector<double>> pW, pWb, pM;    // W_l [q][c], W^beta_l [q][c], M_l [c][n]
  std::vector<double> pchem0, pfilm;               // [S][chan_0], [S * S][2 sum chan]
  size_t psz = 0, bC = 0, bQ = 0;                  // a pair table's size; the back-flow's part
  std::vector<Buf<T>> ptabs;

  // per-call storage (kept between calls)
  std::vector<Buf<T>> Abuf, Ibuf, gAbuf, gfeat;    // per layer, per atom
  Buf<int32_t> nact, act;
  Buf<T> dvec;                                     // dE/d(bond vector), 3 per candidate
  Buf<double> eat;
  Buf<T> fgate;                                    // gate factor of every layer and atom, [layer][atom]
  Buf<double> core_c, core_dc;                     // the core switch of an atom's ZBL energy and its derivative
  Buf<T> geo;                                      // per atom: its bonds' geometry (geo_block)
  Buf<int64_t> goff;                               // where each atom's block starts
  struct Work {                                    // one thread's scratch
    Buf<T> x, gx, P, gP, gI, dR, blockA;
    Buf<T> dx, gA0, Qr;                            // d(squash)/d(feature); layer 0's gA before its gate; reverse Q
    double gf0 = 0;                                // dE/d(layer 0's gate factor) at fixed bond sums (the back-flow)
    std::vector<Buf<T>> z, a, s, g;
    Buf<T> dEdg, dEdY;                             // per bond: dE/dg (nbp), dE/dY (nlmp)
    Buf<T> ga, dgeo;                               // an atom's g per bond (nbp), its derivative rows
    Buf<T> G, dG, dGT;                             // layer 0 per neighbour species
    Buf<T> Ie, Q, dQ;                              // back-flow: a bond's message, Q_l and dE/dQ_l
    Buf<T> Pg, dPg, Gs, Pb, gPb;                   // Gram readout
    std::vector<int> bspec, slist;                 // species slot of each bond, species of a slot
  };
  std::vector<Work> work;
  int MB = 48;                                     // atoms per readout block (multiple of MR)
  int mb_max = 48;
  // seconds per stage, summed over calls (thread 0's view): PRE MID LAST BACK FORCE, readout
  double timers[6] = {0, 0, 0, 0, 0, 0};

  void read_block(Reader& rd, Block& K) {
    K.C = rd.ii();
    K.lo = rd.ii();
    K.nbase = rd.ii();
    K.nrows = rd.ii();
    K.nl = rd.ii();
    const int nterms = rd.ii(), nfrows = rd.ii();
    K.Cp = int(round_up(K.C, W));
    K.NV = K.Cp / W;
    if (K.C < 1 || K.lo < 0 || K.nbase < 1 || K.nrows < K.nbase || K.nl < 0 || nterms < 0 || nfrows < 0)
      throw std::runtime_error("model: bad block sizes");
    if (K.NV > MAXNV || K.nl > MAXNL) throw std::runtime_error("model: too many channels");
    K.terms.resize(nterms);
    for (auto& tm : K.terms) {
      tm.a = rd.ii();
      tm.b = rd.ii();
      tm.o = rd.ii();
      tm.cg = T(rd.d());
      if (tm.a < 0 || tm.b < 0 || tm.a >= tm.o || tm.b >= tm.o || tm.o >= K.nrows || tm.o < K.nbase)
        throw std::runtime_error("model: bad product term");
    }
    std::stable_sort(K.terms.begin(), K.terms.end(), [](const Term& x, const Term& y) { return x.o < y.o; });
    for (int k = 0; k < nterms; ++k)
      if (k == 0 || K.terms[k].o != K.terms[k - 1].o) K.grp.push_back(k);
    K.grp.push_back(nterms);
    // uses of each row as an input, for the backward (a symmetric term twice)
    std::vector<std::vector<Use>> by(K.nrows);
    for (const Term& tm : K.terms) {
      by[tm.a].push_back({tm.o, tm.b, tm.cg});
      by[tm.b].push_back({tm.o, tm.a, tm.cg});
    }
    K.use_off.push_back(0);
    for (auto& u : by) {
      K.uses.insert(K.uses.end(), u.begin(), u.end());
      K.use_off.push_back(int(K.uses.size()));
    }
    K.frows.resize(nfrows);
    for (auto& f : K.frows) {
      f = rd.ii();
      if (f < 0 || f >= K.nrows) throw std::runtime_error("model: bad feature row");
    }
  }

  void read_messages(Reader& rd, Layer& Ly, const Layer& Pv) {
    const Block& K = Ly.B[0];
    Ly.nI = rd.ii();
    const int nred = rd.ii(), nyt = rd.ii();
    if (Ly.nI < 1 || nred < 0 || nyt < 0) throw std::runtime_error("model: bad message sizes");
    Ly.reduce_blk.resize(nred);
    Ly.reduce_from.resize(nred);
    Ly.reduce_to.resize(nred);
    std::vector<double> wts;
    size_t n1 = 0, n2 = 0;
    for (int f = 0; f < nred; ++f) {
      const int b = rd.ii();
      Ly.reduce_blk[f] = b;
      Ly.reduce_from[f] = rd.ii();
      Ly.reduce_to[f] = rd.ii();
      if (b < 0 || b >= int(Pv.B.size()) || (f && b < Ly.reduce_blk[f - 1]) || Ly.reduce_from[f] < 0 ||
          Ly.reduce_from[f] >= Pv.B[b].nrows || Ly.reduce_to[f] < 0 || Ly.reduce_to[f] >= Ly.nI)
        throw std::runtime_error("model: bad reduction row");
      Ly.red_off.push_back(n1);
      Ly.redT_off.push_back(n2);
      n1 += size_t(Pv.B[b].C) * K.Cp;
      n2 += size_t(K.C) * Pv.B[b].Cp;
      for (int k = 0; k < Pv.B[b].C * K.C; ++k) wts.push_back(rd.d());
    }
    Ly.redW.zeros(std::max<size_t>(n1, 1));
    Ly.redWT.zeros(std::max<size_t>(n2, 1));
    size_t at = 0;
    for (int f = 0; f < nred; ++f) {
      const Block& Kp = Pv.B[Ly.reduce_blk[f]];
      for (int c = 0; c < Kp.C; ++c)
        for (int k = 0; k < K.C; ++k) {
          const T w = T(wts[at++]);
          Ly.redW[Ly.red_off[f] + size_t(c) * K.Cp + k] = w;
          Ly.redWT[Ly.redT_off[f] + size_t(k) * Kp.Cp + c] = w;
        }
    }
    // rows of one source block with the same weights (parity, l, path; M differs) form a group
    Ly.rg_off.assign(1, 0);
    for (size_t b = 0; b < Pv.B.size(); ++b) {
      Ly.rg_blk.push_back(int(Ly.rg_off.size()) - 1);
      const size_t wsz = size_t(Pv.B[b].C) * K.Cp;
      std::vector<int> rep;
      std::vector<std::vector<int>> mem;
      for (int f = 0; f < nred; ++f) {
        if (Ly.reduce_blk[f] != int(b)) continue;
        size_t g = 0;
        for (; g < rep.size(); ++g)
          if (mem[g].size() < size_t(MAXG) && std::memcmp(Ly.redW.data() + Ly.red_off[rep[g]],
                                                          Ly.redW.data() + Ly.red_off[f], wsz * sizeof(T)) == 0)
            break;
        if (g == rep.size()) {
          rep.push_back(f);
          mem.emplace_back();
        }
        mem[g].push_back(f);
      }
      for (auto& m : mem) {
        Ly.rg_f.insert(Ly.rg_f.end(), m.begin(), m.end());
        Ly.rg_off.push_back(int(Ly.rg_f.size()));
      }
    }
    Ly.rg_blk.push_back(int(Ly.rg_off.size()) - 1);
    Ly.yt.resize(nyt);
    for (auto& y : Ly.yt) {
      y.lm = rd.ii();
      y.r = rd.ii();
      y.o = rd.ii();
      y.cg = T(rd.d());
      if (y.lm < 0 || y.lm >= n_lm_feat || y.r < 0 || y.r >= Ly.nI || y.o < 0 || y.o >= K.nbase)
        throw std::runtime_error("model: bad edge coupling term");
      y.deg = deg[y.lm];
    }
    std::stable_sort(Ly.yt.begin(), Ly.yt.end(), [](const YTerm& x, const YTerm& y) { return x.o < y.o; });
  }

  void read_gram(Reader& rd) {
    rd.tag(1400);
    const Block& K = lay[L - 1].B[0];
    Gram& g = gram;
    g.r = rd.ii();
    g.rb = rd.ii();
    if (g.r < 1 || g.rb < 1 || K.nrows != K.nbase) throw std::runtime_error("model: bad Gram readout");
    g.rp = int(round_up(g.r, W));
    if (g.rp > K.Cp) throw std::runtime_error("model: Gram rank above the channels");
    std::vector<std::vector<double>> Gd(lmax_feat + 1), Bd(lmax_feat + 1);
    for (int l = 0; l <= lmax_feat; ++l) {
      const int nf = rd.ii();
      g.rows.emplace_back(size_t(nf) * (2 * l + 1));
      for (auto& x : g.rows.back()) {
        x = rd.ii();
        if (x < 0 || x >= K.nbase) throw std::runtime_error("model: bad Gram row");
      }
      for (int k = 0; k < nf * K.C * g.r; ++k) Gd[l].push_back(rd.d());
      for (int k = 0; k < nf * K.C * g.rb; ++k) Bd[l].push_back(rd.d());
    }
    read_block(rd, g.bis);
    const Block& Bk = g.bis;
    if (Bk.C != g.rb || Bk.nbase != n_lm_feat || Bk.nl) throw std::runtime_error("model: bad bispectrum");
    for (int l = 0; l <= lmax_feat; ++l) {
      const int nf = int(g.rows[l].size()) / (2 * l + 1);
      g.G.emplace_back();
      g.GT.emplace_back();
      g.Bm.emplace_back();
      g.BT.emplace_back();
      g.G[l].zeros(std::max<size_t>(size_t(nf) * K.C * g.rp, 1));
      g.GT[l].zeros(std::max<size_t>(size_t(nf) * g.r * K.Cp, 1));
      g.Bm[l].zeros(std::max<size_t>(size_t(nf) * K.C * Bk.Cp, 1));
      g.BT[l].zeros(std::max<size_t>(size_t(nf) * g.rb * K.Cp, 1));
      for (int f = 0; f < nf; ++f)
        for (int c = 0; c < K.C; ++c) {
          for (int k = 0; k < g.r; ++k) {
            const T w = T(Gd[l][(size_t(f) * K.C + c) * g.r + k]);
            g.G[l][(size_t(f) * K.C + c) * g.rp + k] = w;
            g.GT[l][(size_t(f) * g.r + k) * K.Cp + c] = w;
          }
          for (int k = 0; k < g.rb; ++k) {
            const T w = T(Bd[l][(size_t(f) * K.C + c) * g.rb + k]);
            g.Bm[l][(size_t(f) * K.C + c) * Bk.Cp + k] = w;
            g.BT[l][(size_t(f) * g.rb + k) * K.Cp + c] = w;
          }
        }
      g.poff.push_back(g.psz);
      g.psz += size_t(2 * l + 1) * g.rp;
    }
    g.npair = g.r * (g.r + 1) / 2;
  }

  void read_pairs(Reader& rd) {
    rd.tag(1500);
    nbf = rd.ii();
    const int Cm = lay[1].B[0].C;
    if (nbf < 1 || nbf > lmax_feat + 1 || lay[1].nI != nbf * nbf) throw std::runtime_error("model: bad back-flow");
    int tot = 0;
    for (int l = 0; l <= lmax; ++l) {
      chan.push_back(rd.ii());
      chan_off.push_back(tot);
      tot += chan.back();
      if (chan.back() < 1 || chan.back() > chan[0]) throw std::runtime_error("model: bad channels");
    }
    for (int l = 0; l <= lmax; ++l) {
      pW.emplace_back(size_t(nb) * chan[l]);
      for (auto& x : pW.back()) x = rd.d();
    }
    pchem0.resize(size_t(S) * chan[0]);
    for (auto& x : pchem0) x = rd.d();
    for (int l = 0; l <= lmax; ++l) {
      pWb.emplace_back(size_t(nb) * chan[l]);
      for (auto& x : pWb.back()) x = rd.d();
    }
    pfilm.resize(size_t(S) * S * 2 * tot);
    for (auto& x : pfilm) x = rd.d();
    for (int l = 0; l < nbf; ++l) {
      pM.emplace_back(size_t(chan[l]) * Cm);
      for (auto& x : pM.back()) x = rd.d();
    }
    // the parts of a pair table
    psz = 0;
    for (Block& K : lay[0].B) {
      for (int l = 0; l < K.nl; ++l)
        if (K.lo + K.C > chan[l]) throw std::runtime_error("model: bad pair block");
      K.pC = psz;
      psz += size_t(K.nl) * nb * K.Cp;
      K.pQ = psz;
      psz += size_t(K.nl) * K.C * nbp;
    }
    size_t mC = 0, mQ = 0;
    for (int t = 1; t < L; ++t) {
      Block& K = lay[t].B[0];
      for (int l = 0; l < K.nl; ++l)
        if (K.C != Cm || K.C > chan[l]) throw std::runtime_error("model: bad pair block");
      if (!mC) {
        mC = psz;
        psz += size_t(K.nl) * nb * K.Cp;
        mQ = psz;
        psz += size_t(K.nl) * K.C * nbp;
      }
      K.pC = mC;
      K.pQ = mQ;
    }
    Layer& L1 = lay[1];
    L1.bf = true;
    L1.ytr = L1.yt;
    std::stable_sort(L1.ytr.begin(), L1.ytr.end(), [](const YTerm& x, const YTerm& y) { return x.r < y.r; });
    L1.yr_off.assign(L1.nI + 1, 0);
    for (const YTerm& y : L1.ytr) ++L1.yr_off[y.r + 1];
    for (int r = 0; r < L1.nI; ++r) L1.yr_off[r + 1] += L1.yr_off[r];
    bC = psz;
    psz += size_t(nbf) * nb * lay[1].B[0].Cp;
    bQ = psz;
    psz += size_t(nbf) * Cm * nbp;
    ptabs.resize(size_t(S) * S);
  }

  void load(Reader& rd) {
    rd.tag(MAGIC);
    if (rd.i() != VERSION) throw std::runtime_error("model: stream version mismatch");
    S = rd.ii();
    L = rd.ii();
    lmax = rd.ii();
    lmax_feat = rd.ii();
    nb = rd.ii();
    cheb_p = rd.ii();
    zbl = rd.i() != 0;
    const int n_hidden = rd.ii();
    if (L < 2 || lmax > LMAX_SH || lmax_feat > lmax || nb < 1 || nb > 64 || n_hidden < 1 || cheb_p < 1)
      throw std::runtime_error("model: unsupported sizes");
    n_lm = (lmax + 1) * (lmax + 1);
    n_lm_feat = (lmax_feat + 1) * (lmax_feat + 1);
    nlmp = int(round_up(n_lm, W));
    nbp = int(round_up(nb, W));
    if (nbp / W > MAXQ || nlmp / W > MAXQ) throw std::runtime_error("model: too many radial functions or harmonics");
    for (int l = 0; l <= lmax; ++l)
      for (int m = -l; m <= l; ++m) deg.push_back(l);
    rcut = rd.d();
    zbl_factor = rd.d();
    scale = rd.d();
    build_harmonics();

    lay.resize(L);
    for (int t = 0; t < L; ++t) {
      Layer& Ly = lay[t];
      rd.tag(1000 + t);
      const int nblk = rd.ii();
      if (nblk < 1 || nblk > LMAX_SH + 1 || (t > 0 && nblk != 1)) throw std::runtime_error("model: bad blocks");
      Ly.B.resize(nblk);
      int covered = -1;
      for (Block& K : Ly.B) {
        read_block(rd, K);
        K.aoff = Ly.asz;
        Ly.asz += size_t(K.nbase) * K.Cp;
        K.goff = Ly.gsz;
        Ly.gsz += K.frows.size() * K.Cp;
        if (t == 0) {
          if (K.nbase != K.nl * K.nl)
            throw std::runtime_error("model: layer 0 must have one row per harmonic");
          K.lnew = covered;
          covered = std::max(covered, K.nl - 1);
        }
      }
      if (t == 0 && covered != lmax) throw std::runtime_error("model: radial channels of layer 0");
      if (t > 0) {
        if (Ly.B[0].nl != lmax_feat + 1) throw std::runtime_error("model: radial channels of a later layer");
        read_messages(rd, Ly, lay[t - 1]);
      }
    }
    read_gram(rd);
    read_pairs(rd);

    // readout
    rd.tag(2000);
    F = rd.ii();
    int at = 0;
    for (int t = 0; t < L; ++t)
      for (Block& K : lay[t].B) {
        K.foff = at;
        at += int(K.frows.size()) * K.C;
      }
    gram.foff = at;
    at += (lmax_feat + 1) * gram.npair;
    gram.bis.foff = at;
    at += int(gram.bis.frows.size()) * gram.bis.C;
    if (at != F) throw std::runtime_error("model: feature count mismatch");
    F_ld = int(round_up(F, W));
    mlp.resize(n_hidden);
    for (int h = 0; h < n_hidden; ++h) {
      Dense& D = mlp[h];
      D.nin = rd.ii();
      D.nout = rd.ii();
      if (D.nin != (h ? mlp[h - 1].nout : F)) throw std::runtime_error("model: readout sizes");
      D.nout_p = int(round_up(D.nout, NR));
      D.nin_p = int(round_up(D.nin, NR));
      std::vector<double> Wt(size_t(D.nout) * D.nin);
      for (auto& w : Wt) w = rd.d();
      D.Wf.zeros(size_t(D.nout_p) * D.nin);
      for (int n = 0; n < D.nout; ++n)
        for (int k = 0; k < D.nin; ++k)
          D.Wf[(size_t(n / NR) * D.nin + k) * NR + n % NR] = T(Wt[size_t(n) * D.nin + k]);
      D.Wb.zeros(size_t(D.nin_p) * D.nout);
      for (int k = 0; k < D.nout; ++k)
        for (int n = 0; n < D.nin; ++n)
          D.Wb[(size_t(n / NR) * D.nout + k) * NR + n % NR] = T(Wt[size_t(k) * D.nin + n]);
      const int nb_rows = h == 0 ? S : 1;
      D.bias.zeros(size_t(nb_rows) * D.nout_p);
      for (int s = 0; s < nb_rows; ++s)
        for (int n = 0; n < D.nout; ++n) D.bias[size_t(s) * D.nout_p + n] = T(rd.d());
    }
    Hlast = mlp.back().nout;
    head.zeros(size_t(S) * mlp.back().nout_p);
    for (int s = 0; s < S; ++s)
      for (int n = 0; n < Hlast; ++n) head[size_t(s) * mlp.back().nout_p + n] = T(rd.d());
    for (int s = 0; s < S; ++s) e0.push_back(rd.d());
    for (int s = 0; s < S; ++s) zbl_z.push_back(rd.d());
    for (int s = 0; s < S; ++s) zbl_rcov.push_back(rd.d());
    rd.tag(2500);
    gate = rd.i() != 0;
    squash = rd.i() != 0;
    core = rd.i() != 0;
    const double a0 = rd.d(), kappa = rd.d();
    tau = rd.d();
    zc = rd.d();
    if ((gate && !(a0 > 0 && kappa > 0)) || (squash && !(tau > 0)) || (core && !(zc > 0)) || (core && !zbl))
      throw std::runtime_error("model: bad bounds");
    for (int t = 0; gate && t < L; ++t) {
      const double sigma = rd.d();
      int64_t n = 0;
      for (const Block& K : lay[t].B) n += int64_t(K.nbase) * K.C;
      if (!(sigma > 0)) throw std::runtime_error("model: bad bounds");
      gate_c.push_back(a0 / sigma);
      gate_s2.push_back(sigma * kappa * sigma * kappa);
      gate_n.push_back(double(n));
    }
    gate_k2 = a0 * a0 * kappa * kappa;
    for (int c = 0; squash && c < 2 * F; ++c) (c < F ? fmean : fstd).push_back(T(rd.d()));
    if (rd.pi < rd.nI && rd.I[rd.pi] == 3000) {
      rd.tag(3000);
      for (int k = 0; k < S * S; ++k) vc_scale.push_back(rd.d());
      for (int k = 0; k < S * S; ++k) vc_rc2.push_back(rd.d());
    }
    rd.tag(MAGIC);
    if (rd.pi != rd.nI || rd.pd != rd.nD) throw std::runtime_error("model: streams longer than the model");
    if (mlp[0].nout >= 512) mb_max = 96;           // wide layers: fewer passes over their weights
    if (const char* e = std::getenv("ILMLIP_CPU_BLOCK"))
      mb_max = std::min(504, std::max(MR, int(round_up(std::atoi(e), MR))));
  }

  struct Input {
    int n;
    const double* pos;
    const int32_t* mu;
    const int64_t* off;                            // n + 1 candidate offsets (sorted by atom)
    const int32_t* cj;                             // neighbour of each candidate
    const double* csh;                             // its shift (3 per candidate)
    const int32_t* rev;                            // the reverse candidate
  };
  // the pair tables of every pair of species in the structure
  void build_pairs(const Input& in) {
    std::vector<char> seen(S, 0);
    for (int i = 0; i < in.n; ++i) seen[in.mu[i]] = 1;
    for (int a = 0; a < S; ++a)
      for (int b = 0; b < S && seen[a]; ++b)
        if (seen[b] && !ptabs[size_t(a) * S + b].p) build_pair(a, b);
  }
  void build_pair(int a, int b) {
    const size_t key = size_t(a) * S + b;
    const int tot = chan_off.back() + chan.back(), C0 = chan[0];
    std::vector<std::vector<double>> Fl(lmax + 1);
    for (int l = 0; l <= lmax; ++l) {
      const int ch = chan[l];
      Fl[l].resize(size_t(nb) * ch);
      for (int q = 0; q < nb; ++q)
        for (int c = 0; c < ch; ++c) {
          const double* fr = pfilm.data() + key * 2 * tot + chan_off[l] + c;
          const double w = pW[l][size_t(q) * ch + c] * (1.0 + fr[0]) + pWb[l][size_t(q) * ch + c] * fr[tot];
          Fl[l][size_t(q) * ch + c] = w * pchem0[size_t(b) * C0 + c];
        }
    }
    Buf<T> tab;
    tab.zeros(psz);
    auto put = [&](const Block& K, size_t pC, size_t pQ, int lo) {
      for (int l = 0; l < K.nl; ++l)
        for (int q = 0; q < nb; ++q)
          for (int c = 0; c < K.C; ++c) {
            const T w = T(Fl[l][size_t(q) * chan[l] + lo + c]);
            tab[pC + (size_t(l) * nb + q) * K.Cp + c] = w;
            tab[pQ + (size_t(l) * K.C + c) * nbp + q] = w;
          }
    };
    for (const Block& K : lay[0].B) put(K, K.pC, K.pQ, K.lo);
    for (int t = 1; t < L; ++t) put(lay[t].B[0], lay[t].B[0].pC, lay[t].B[0].pQ, 0);
    const Block& K1 = lay[1].B[0];
    for (int l = 0; l < nbf; ++l)
      for (int q = 0; q < nb; ++q)
        for (int n = 0; n < K1.C; ++n) {
          double v = 0.0;
          for (int c = 0; c < chan[l]; ++c) v += Fl[l][size_t(q) * chan[l] + c] * pM[l][size_t(c) * K1.C + n];
          tab[bC + (size_t(l) * nb + q) * K1.Cp + n] = T(v);
          tab[bQ + (size_t(l) * K1.C + n) * nbp + q] = T(v);
        }
    ptabs[key] = std::move(tab);
  }
  const T* ptab(int a, int b) const { return ptabs[size_t(a) * S + b].data(); }

  // rows nbase.. of P from its base rows, one register accumulator per output row
  template <int NV_> static void products_forward(const Block& Ly, T* P) {
    constexpr int NM = NV_ ? NV_ : MAXNV;
    const int nv = NV_ ? NV_ : Ly.NV, Cp = Ly.Cp;
    for (size_t g = 0; g + 1 < Ly.grp.size(); ++g) {
      V acc[NM];
      for (int k = 0; k < nv; ++k) acc[k] = V{};
      for (int t = Ly.grp[g]; t < Ly.grp[g + 1]; ++t) {
        const Term& tm = Ly.terms[t];
        const T* pa = P + size_t(tm.a) * Cp;
        const T* pb = P + size_t(tm.b) * Cp;
        const V c = splat(tm.cg);
        for (int k = 0; k < nv; ++k) acc[k] += c * (ld(pa + k * W) * ld(pb + k * W));
      }
      T* po = P + size_t(Ly.terms[Ly.grp[g]].o) * Cp;
      for (int k = 0; k < nv; ++k) st(po + k * W, acc[k]);
    }
  }
  // gP of every row (each row's own gradient on entry) -> gP of the base rows; rows from
  // the top, each gathering what its uses send back
  template <int NV_> static void products_backward(const Block& Ly, const T* P, T* gP) {
    constexpr int NM = NV_ ? NV_ : MAXNV;
    const int nv = NV_ ? NV_ : Ly.NV, Cp = Ly.Cp;
    for (int x = Ly.nrows - 1; x >= 0; --x) {
      const int u0 = Ly.use_off[x], u1 = Ly.use_off[x + 1];
      if (u0 == u1) continue;
      T* gx = gP + size_t(x) * Cp;
      V acc[NM];
      for (int k = 0; k < nv; ++k) acc[k] = ld(gx + k * W);
      for (int u = u0; u < u1; ++u) {
        const Use& us = Ly.uses[u];
        const T* go = gP + size_t(us.o) * Cp;
        const T* po = P + size_t(us.other) * Cp;
        const V c = splat(us.cg);
        for (int k = 0; k < nv; ++k) acc[k] += c * (ld(go + k * W) * ld(po + k * W));
      }
      for (int k = 0; k < nv; ++k) st(gx + k * W, acc[k]);
    }
  }
  // message of layer t (>= 1) from block Pv of layer t-1, added; the rows of a group share
  // their weights, so the group is reduced in one pass over the weights
  template <int NV_, int G_>
  void reduce_group_fwd(const Layer& Ly, const Block& Pv, const int* mem, int gsz, const T* Pprev, T* I) const {
    const Block& K = Ly.B[0];
    constexpr int NM = NV_ ? NV_ : MAXNV, GM = G_ ? G_ : MAXG;
    const int nv = NV_ ? NV_ : K.NV, ng = G_ ? G_ : gsz, Cp = K.Cp;
    V acc[GM][NM];
    const T* src[GM];
    for (int m = 0; m < ng; ++m) {
      src[m] = Pprev + size_t(Ly.reduce_from[mem[m]]) * Pv.Cp;
      for (int k = 0; k < nv; ++k) acc[m][k] = V{};
    }
    const T* w = Ly.redW.data() + Ly.red_off[mem[0]];
    for (int c = 0; c < Pv.C; ++c) {
      V wk[NM];
      for (int k = 0; k < nv; ++k) wk[k] = ld(w + size_t(c) * Cp + k * W);
      for (int m = 0; m < ng; ++m) {
        const V sc = splat(src[m][c]);
        for (int k = 0; k < nv; ++k) acc[m][k] += sc * wk[k];
      }
    }
    for (int m = 0; m < ng; ++m) {
      T* d = I + size_t(Ly.reduce_to[mem[m]]) * Cp;
      for (int k = 0; k < nv; ++k) st(d + k * W, ld(d + k * W) + acc[m][k]);
    }
  }
  // gI (w.r.t. the message before chem) -> gP of block Pv of layer t-1 (added), per weight group
  template <int NVP_, int G_>
  void reduce_group_bwd(const Layer& Ly, const Block& Pv, const int* mem, int gsz, const T* gI, T* gP) const {
    const Block& K = Ly.B[0];
    constexpr int NM = NVP_ ? NVP_ : MAXNV, GM = G_ ? G_ : MAXG;
    const int nv = NVP_ ? NVP_ : Pv.NV, ng = G_ ? G_ : gsz, Cpp = Pv.Cp;
    V acc[GM][NM];
    const T* g[GM];
    for (int m = 0; m < ng; ++m) {
      g[m] = gI + size_t(Ly.reduce_to[mem[m]]) * K.Cp;
      const T* d = gP + size_t(Ly.reduce_from[mem[m]]) * Cpp;
      for (int k = 0; k < nv; ++k) acc[m][k] = ld(d + k * W);
    }
    const T* w = Ly.redWT.data() + Ly.redT_off[mem[0]];
    for (int c = 0; c < K.C; ++c) {
      V wk[NM];
      for (int k = 0; k < nv; ++k) wk[k] = ld(w + size_t(c) * Cpp + k * W);
      for (int m = 0; m < ng; ++m) {
        const V sc = splat(g[m][c]);
        for (int k = 0; k < nv; ++k) acc[m][k] += sc * wk[k];
      }
    }
    for (int m = 0; m < ng; ++m) {
      T* d = gP + size_t(Ly.reduce_from[mem[m]]) * Cpp;
      for (int k = 0; k < nv; ++k) st(d + k * W, acc[m][k]);
    }
  }
  template <typename F> static void dispatch_g(int n, F&& f) {
    switch (n) {
      case 1: f(std::integral_constant<int, 1>()); return;
      case 3: f(std::integral_constant<int, 3>()); return;
      case 5: f(std::integral_constant<int, 5>()); return;
      default: f(std::integral_constant<int, 0>()); return;
    }
  }
  void products_forward(const Block& Ly, T* P) const {
    dispatch(Ly.NV, [&](auto nv) { products_forward<decltype(nv)::value>(Ly, P); });
  }
  void products_backward(const Block& Ly, const T* P, T* gP) const {
    dispatch(Ly.NV, [&](auto nv) { products_backward<decltype(nv)::value>(Ly, P, gP); });
  }
  // I of layer t += the reduction of block b of layer t-1 (rows P)
  void reduce_add(int t, int b, const T* Pprev, T* I) const {
    const Layer& Ly = lay[t];
    const Block& Pv = lay[t - 1].B[b];
    for (int g = Ly.rg_blk[b]; g < Ly.rg_blk[b + 1]; ++g) {
      const int* mem = Ly.rg_f.data() + Ly.rg_off[g];
      const int gsz = Ly.rg_off[g + 1] - Ly.rg_off[g];
      dispatch(Ly.B[0].NV, [&](auto nv) {
        dispatch_g(gsz, [&](auto ng) {
          this->template reduce_group_fwd<decltype(nv)::value, decltype(ng)::value>(Ly, Pv, mem, gsz, Pprev, I);
        });
      });
    }
  }
  void reduce_bwd(int t, int b, const T* gI, T* gPprev) const {
    const Layer& Ly = lay[t];
    const Block& Pv = lay[t - 1].B[b];
    for (int g = Ly.rg_blk[b]; g < Ly.rg_blk[b + 1]; ++g) {
      const int* mem = Ly.rg_f.data() + Ly.rg_off[g];
      const int gsz = Ly.rg_off[g + 1] - Ly.rg_off[g];
      dispatch(Pv.NV, [&](auto nv) {
        dispatch_g(gsz, [&](auto ng) {
          this->template reduce_group_bwd<decltype(nv)::value, decltype(ng)::value>(Ly, Pv, mem, gsz, gI, gPprev);
        });
      });
    }
  }
  // load-time, not vectorised, like the polynomial code above
  __attribute__((optimize("no-tree-vectorize"))) void build_harmonics() {
    static Monomials Mfull(LMAX_SH);
    Poly::M = &Mfull;
    Poly X = poly_zero(), Yp = poly_zero(), Z = poly_zero();
    X.c[Mfull.index[1][0][0]] = 1.0;
    Yp.c[Mfull.index[0][1][0]] = 1.0;
    Z.c[Mfull.index[0][0][1]] = 1.0;
    std::vector<Poly> out((LMAX_SH + 1) * (LMAX_SH + 1));
    harmonics<Poly>(lmax, X, Yp, Z, out.data());
    const Monomials M(lmax);                       // the monomials evaluated per bond
    n_mono = M.n;
    mono_parent.assign(M.parent, M.parent + M.n);
    mono_axis.assign(M.axis, M.axis + M.n);
    auto local = [&](int A, int B, int C) { return M.index[A][B][C]; };
    sh_off.assign(1, 0);
    dsh_off.assign(1, 0);
    for (int lm = 0; lm < n_lm; ++lm) {
      double big = 0.0;
      for (int i = 0; i < Mfull.n; ++i) big = std::max(big, std::fabs(out[lm].c[i]));
      for (int i = 0; i < Mfull.n; ++i) {
        const double cf = out[lm].c[i];
        if (std::fabs(cf) <= 1e-14 * big) continue;
        if (Mfull.a[i] + Mfull.b[i] + Mfull.c[i] > lmax) throw std::runtime_error("harmonics: degree");
        sh_idx.push_back(local(Mfull.a[i], Mfull.b[i], Mfull.c[i]));
        sh_c.push_back(cf);
      }
      sh_off.push_back(int(sh_idx.size()));
      for (int ax = 0; ax < 3; ++ax) {
        for (int i = 0; i < Mfull.n; ++i) {
          const double cf = out[lm].c[i];
          if (std::fabs(cf) <= 1e-14 * big) continue;
          int p[3] = {Mfull.a[i], Mfull.b[i], Mfull.c[i]};
          if (p[ax] == 0) continue;
          const double f = cf * p[ax];
          --p[ax];
          dsh_idx.push_back(local(p[0], p[1], p[2]));
          dsh_c.push_back(f);
        }
        dsh_off.push_back(int(dsh_idx.size()));
      }
    }
    Poly::M = nullptr;
  }

  // Per atom, the geometry of its bonds as rows over the bonds (nbs = bonds rounded up to W):
  // r, u_x, u_y, u_z, g[q], Y[lm].  The derivative rows, dg/dr[q], dY[lm][axis], exist only where
  // a thread needs them (bond_deriv): the blocks of all atoms do not fit in the caches.
  int geo_rows() const { return 4 + nb + n_lm; }
  size_t geo_block(int nbs) const { return size_t(geo_rows()) * nbs; }
  int R_G() const { return 4; }
  int R_Y() const { return 4 + nb; }

  // W bonds at once, lanes = bonds (bond vectors vx, vy, vz): rows at out (row stride nbs), the
  // block's rows or with DERIV the derivative rows
  template <bool DERIV> void geometry_lanes(V vx, V vy, V vz, T* out, int nbs) const {
    const T rc = T(rcut), eps = T(1e-8), one = T(1);
    const V rr = vsqrt(vx * vx + vy * vy + vz * vz);
    const V r = rr > eps ? rr : splat(eps);
    const V ux = vx / r, uy = vy / r, uz = vz / r;
    // u = min(r / rc, 1): from rc on the envelope and its slope are zero, and so is every g
    const V uu = r / rc;
    const V u = uu < one ? uu : splat(one);
    const V xi = T(2) * u - one;
    const V dxi = splat(T(2) / rc);
    // envelope (1 - u)^3 sum_k (k+1)(k+2)/2 u^k (k < p), in Horner's form (DimeNet's polynomial
    // envelope, Gasteiger et al. 2020, factored)
    const int p = cheb_p;
    V qv = splat(T(0.5 * p * (p + 1))), dq{};
    for (int c = p - 2; c >= 0; --c) {
      dq = dq * u + qv;
      qv = qv * u + T(0.5 * (c + 1) * (c + 2));
    }
    const V wv = one - u;
    const V f = wv * wv * wv * qv;
    const V df = (wv * wv * (wv * dq - T(3) * qv)) / rc;
    V t0 = splat(one), t1 = xi, d0{}, d1 = splat(one);
    if constexpr (!DERIV) {
      st(out, r);
      st(out + nbs, ux);
      st(out + 2 * nbs, uy);
      st(out + 3 * nbs, uz);
    }
    // g_q = f T_q(xi), q = 1..nb, by the three-term recurrence, with dg_q/dr
    for (int q = 0; q < nb; ++q) {
      if constexpr (DERIV)
        st(out + size_t(q) * nbs, d1 * dxi * f + t1 * df);
      else
        st(out + size_t(R_G() + q) * nbs, t1 * f);
      const V t2 = T(2) * xi * t1 - t0, d2 = T(2) * t1 + T(2) * xi * d1 - d0;
      t0 = t1;
      t1 = t2;
      d0 = d1;
      d1 = d2;
    }
    // harmonics: the monomials of (u_x, u_y, u_z), then the polynomials and their derivatives
    V m[NMONO];
    m[0] = splat(one);
    for (int i = 1; i < n_mono; ++i) m[i] = m[mono_parent[i]] * (mono_axis[i] == 0 ? ux : mono_axis[i] == 1 ? uy : uz);
    if constexpr (DERIV) {
      for (int j = 0; j < 3 * n_lm; ++j) {
        V acc{};
        for (int t = dsh_off[j]; t < dsh_off[j + 1]; ++t) acc += T(dsh_c[t]) * m[dsh_idx[t]];
        st(out + size_t(nb + j) * nbs, acc);
      }
    } else {
      for (int lm = 0; lm < n_lm; ++lm) {
        V acc{};
        for (int t = sh_off[lm]; t < sh_off[lm + 1]; ++t) acc += T(sh_c[t]) * m[sh_idx[t]];
        st(out + size_t(R_Y() + lm) * nbs, acc);
      }
    }
  }
  // W bonds of atom i from bond b0, spare lanes: any bond inside rcut; vcut: scaled, sc the factors
  void bond_vectors(const Input& in, int i, int nbond, const int32_t* bonds, int b0, V& vx, V& vy, V& vz, V& sc) const {
    for (int k = 0; k < W; ++k) {
      const int b = b0 + k;
      if (b < nbond) {
        const int c = bonds[b], j = in.cj[c];
        const double* sh = in.csh + 3 * size_t(c);
        vx[k] = T(in.pos[3 * j] - in.pos[3 * i] + sh[0]);
        vy[k] = T(in.pos[3 * j + 1] - in.pos[3 * i + 1] + sh[1]);
        vz[k] = T(in.pos[3 * j + 2] - in.pos[3 * i + 2] + sh[2]);
      } else {
        vx[k] = T(0.5 * rcut);
        vy[k] = T(0);
        vz[k] = T(0);
      }
    }
    sc = splat(T(1));
    for (int k = 0; k < W && b0 + k < nbond && !vc_scale.empty(); ++k)
      sc[k] = T(vc_scale[size_t(in.mu[i]) * S + in.mu[in.cj[bonds[b0 + k]]]]);
    vx *= sc;
    vy *= sc;
    vz *= sc;
  }
  // the geometry of atom i's bonds into its block
  void bond_block(const Input& in, int i, int nbond, const int32_t* bonds, T* blk) const {
    const int nbs = int(round_up(nbond, W));
    for (int b0 = 0; b0 < nbs; b0 += W) {
      V vx, vy, vz, sc;
      bond_vectors(in, i, nbond, bonds, b0, vx, vy, vz, sc);
      geometry_lanes<false>(vx, vy, vz, blk + b0, nbs);
      if (!vc_scale.empty()) st(blk + b0, ld(blk + b0) / sc);
    }
  }
  // the derivative rows of atom i's bonds, dg/dr[q] then dY[lm][axis], into w.dgeo
  void bond_deriv(Work& w, const Input& in, int i, int nbond, const int32_t* bonds) const {
    const int nbs = int(round_up(nbond, W));
    for (int b0 = 0; b0 < nbs; b0 += W) {
      V vx, vy, vz, sc;
      bond_vectors(in, i, nbond, bonds, b0, vx, vy, vz, sc);
      T* out = w.dgeo.data() + b0;
      geometry_lanes<true>(vx, vy, vz, out, nbs);
      if (!vc_scale.empty()) {
        for (int q = 0; q < nb; ++q) {
          T* d = out + size_t(q) * nbs;
          st(d, ld(d) * sc);
        }
      }
    }
  }
  // each bond's slot in the list of its atom's neighbour species
  static void species_map(Work& w, const Input& in, int nbond, const int32_t* bonds) {
    w.slist.clear();
    for (int b = 0; b < nbond; ++b) {
      const int mu = in.mu[in.cj[bonds[b]]];
      int s = 0;
      while (s < int(w.slist.size()) && w.slist[s] != mu) ++s;
      if (s == int(w.slist.size())) w.slist.push_back(mu);
      w.bspec[b] = s;
    }
  }

  // dE/d(bond vector) of every bond from its dE/dg and dE/dY (added to dvec)
  void finish_bonds(Work& w, const Input& in, int i, const T* blk, int nbs, int nbond, const int32_t* bonds) {
    bond_deriv(w, in, i, nbond, bonds);
    const T* dg = w.dgeo.data();
    const T* dy = dg + size_t(nb) * nbs;
    for (int b = 0; b < nbond; ++b) {
      const T* eg = w.dEdg.data() + size_t(b) * nbp;
      const T* ey = w.dEdY.data() + size_t(b) * nlmp;
      double dEdr = 0.0, du0 = 0.0, du1 = 0.0, du2 = 0.0;
      for (int q = 0; q < nb; ++q) dEdr += double(eg[q]) * double(dg[size_t(q) * nbs + b]);
      for (int lm = 0; lm < n_lm; ++lm) {
        const double e = double(ey[lm]);
        du0 += e * double(dy[size_t(3 * lm) * nbs + b]);
        du1 += e * double(dy[size_t(3 * lm + 1) * nbs + b]);
        du2 += e * double(dy[size_t(3 * lm + 2) * nbs + b]);
      }
      const double r = blk[b], ux = blk[nbs + b], uy = blk[2 * nbs + b], uz = blk[3 * nbs + b];
      const double pr = du0 * ux + du1 * uy + du2 * uz;
      T* out = dvec.data() + 3 * size_t(bonds[b]);
      out[0] += T(dEdr * ux + (du0 - pr * ux) / r);
      out[1] += T(dEdr * uy + (du1 - pr * uy) / r);
      out[2] += T(dEdr * uz + (du2 - pr * uz) / r);
    }
  }

  template <int NQ_> void layer0_G(Work& w, const T* blk, int nbs, int nbond) const {
    constexpr int QM = NQ_ ? NQ_ : MAXQ;
    const int nq = NQ_ ? NQ_ : nbp / W, gsz = n_lm * nbp, ry = R_Y();
    T* ga = w.ga.data();
    for (int q = 0; q < nb; ++q)
      for (int b = 0; b < nbond; ++b) ga[size_t(b) * nbp + q] = blk[size_t(R_G() + q) * nbs + b];
    for (int b = 0; b < nbond; ++b)
      for (int q = nb; q < nbp; ++q) ga[size_t(b) * nbp + q] = T(0);
    T* G = w.G.data();
    std::memset(G, 0, sizeof(T) * w.slist.size() * gsz);
    for (int b = 0; b < nbond; ++b) {
      T* Gs = G + size_t(w.bspec[b]) * gsz;
      const T* gq = ga + size_t(b) * nbp;
      V gv[QM];
      for (int k = 0; k < nq; ++k) gv[k] = ld(gq + k * W);
      for (int lm = 0; lm < n_lm; ++lm) {
        const V y = splat(blk[size_t(ry + lm) * nbs + b]);
        T* gl = Gs + size_t(lm) * nbp;
        for (int k = 0; k < nq; ++k) st(gl + k * W, ld(gl + k * W) + y * gv[k]);
      }
    }
  }
  // block K of layer 0, centre species mu: A[lm][:] = sum_s sum_q G_s[lm][q] tab(mu, s)[l][q][:] with
  // G_s = sum over s's bonds of Y g
  template <int NV_> void layer0_block(const Block& K, const Work& w, int mu, T* A) const {
    constexpr int NM = NV_ ? NV_ : MAXNV;
    const int nv = NV_ ? NV_ : K.NV, Cp = K.Cp, nlm = K.nl * K.nl;
    const int ns = int(w.slist.size()), gsz = n_lm * nbp;
    const T* G = w.G.data();
    std::memset(A, 0, sizeof(T) * size_t(nlm) * Cp);
    for (int s = 0; s < ns; ++s) {
      const T* Gs = G + size_t(s) * gsz;
      const T* W0 = ptab(mu, w.slist[s]) + K.pC;
      for (int lm = 0; lm < nlm; ++lm) {
        const T* wl = W0 + size_t(deg[lm]) * nb * Cp;
        const T* gl = Gs + size_t(lm) * nbp;
        V a0[NM], a1[NM];
        for (int k = 0; k < nv; ++k) a0[k] = a1[k] = V{};
        int q = 0;
        for (; q + 1 < nb; q += 2) {
          const V s0 = splat(gl[q]), s1 = splat(gl[q + 1]);
          const T* w0 = wl + size_t(q) * Cp;
          for (int k = 0; k < nv; ++k) {
            a0[k] += s0 * ld(w0 + k * W);
            a1[k] += s1 * ld(w0 + Cp + k * W);
          }
        }
        if (q < nb) {
          const V s0 = splat(gl[q]);
          for (int k = 0; k < nv; ++k) a0[k] += s0 * ld(wl + size_t(q) * Cp + k * W);
        }
        T* Al = A + size_t(lm) * Cp;
        for (int k = 0; k < nv; ++k) st(Al + k * W, ld(Al + k * W) + (a0[k] + a1[k]));
      }
    }
  }
  // A of every block of layer 0 (the layer's per-atom rows)
  void layer0_density(Work& w, const T* blk, int nbs, int nbond, int mu, T* A) const {
    dispatch_q(nbp / W, [&](auto nq) { this->template layer0_G<decltype(nq)::value>(w, blk, nbs, nbond); });
    for (const Block& K : lay[0].B)
      dispatch(K.NV, [&](auto nv) { this->template layer0_block<decltype(nv)::value>(K, w, mu, A + K.aoff); });
  }
  // gA of block K -> per species dG_s[lm][:] = sum_c gA[lm][c] tab(mu, s)[pQ + (l C + c) nbp + :],
  // stored for its degrees above K.lnew, added for the others; and dGT, the same as [q][lm]
  template <int NQ_> void layer0_dG(const Block& K, Work& w, int mu, const T* gA) const {
    constexpr int QM = NQ_ ? NQ_ : MAXQ;
    const int Cp = K.Cp, C = K.C, nq = NQ_ ? NQ_ : nbp / W, nlm = K.nl * K.nl;
    const int ns = int(w.slist.size()), gsz = n_lm * nbp, tsz = nb * nlmp;
    T* dG = w.dG.data();
    T* dGT = w.dGT.data();
    for (int s = 0; s < ns; ++s) {
      const T* Q0 = ptab(mu, w.slist[s]) + K.pQ;
      T* dGs = dG + size_t(s) * gsz;
      T* dGTs = dGT + size_t(s) * tsz;
      for (int lm = 0; lm < nlm; ++lm) {
        const T* ga = gA + size_t(lm) * Cp;
        const T* wq = Q0 + size_t(deg[lm]) * C * nbp;
        V a0[QM], a1[QM];
        for (int k = 0; k < nq; ++k) a0[k] = a1[k] = V{};
        int c = 0;
        for (; c + 1 < C; c += 2) {
          const V s0 = splat(ga[c]), s1 = splat(ga[c + 1]);
          const T* w0 = wq + size_t(c) * nbp;
          for (int k = 0; k < nq; ++k) {
            a0[k] += s0 * ld(w0 + k * W);
            a1[k] += s1 * ld(w0 + nbp + k * W);
          }
        }
        if (c < C) {
          const V s0 = splat(ga[c]);
          for (int k = 0; k < nq; ++k) a0[k] += s0 * ld(wq + size_t(c) * nbp + k * W);
        }
        const bool add = deg[lm] <= K.lnew;
        for (int k = 0; k < nq; ++k) {
          V v = a0[k] + a1[k];
          T* d = dGs + size_t(lm) * nbp + k * W;
          if (add) v += ld(d);
          st(d, v);
          for (int e = 0; e < W && k * W + e < nb; ++e) dGTs[size_t(k * W + e) * nlmp + lm] = v[e];
        }
      }
    }
  }
  // every bond's dE/dg and dE/dY (added) from dG: dE/dg += sum_lm Y[lm] dG_s[lm][:],
  // dE/dY[lm] += sum_q g[q] dG_s[lm][q]
  template <int NQ_, int NLV_> void layer0_bonds(Work& w, const T* blk, int nbs, int nbond) const {
    constexpr int QM = NQ_ ? NQ_ : MAXQ, YM = NLV_ ? NLV_ : MAXQ;
    const int nq = NQ_ ? NQ_ : nbp / W, nlv = NLV_ ? NLV_ : nlmp / W;
    const int ns = int(w.slist.size()), gsz = n_lm * nbp, tsz = nb * nlmp, ry = R_Y(), rg = R_G();
    const T* dG = w.dG.data();
    T* dGT = w.dGT.data();
    for (int s = 0; s < ns; ++s)
      for (int q = 0; q < nb; ++q)
        for (int lm = n_lm; lm < nlmp; ++lm) dGT[size_t(s) * tsz + size_t(q) * nlmp + lm] = T(0);
    for (int b = 0; b < nbond; ++b) {
      const int s = w.bspec[b];
      const T* dGs = dG + size_t(s) * gsz;
      const T* dGTs = dGT + size_t(s) * tsz;
      T* eg = w.dEdg.data() + size_t(b) * nbp;
      T* ey = w.dEdY.data() + size_t(b) * nlmp;
      V ag[QM], ay[YM];
      for (int k = 0; k < nq; ++k) ag[k] = ld(eg + k * W);
      for (int lm = 0; lm < n_lm; ++lm) {
        const V y = splat(blk[size_t(ry + lm) * nbs + b]);
        for (int k = 0; k < nq; ++k) ag[k] += y * ld(dGs + size_t(lm) * nbp + k * W);
      }
      for (int k = 0; k < nq; ++k) st(eg + k * W, ag[k]);
      for (int k = 0; k < nlv; ++k) ay[k] = ld(ey + k * W);
      for (int q = 0; q < nb; ++q) {
        const V gq = splat(blk[size_t(rg + q) * nbs + b]);
        for (int k = 0; k < nlv; ++k) ay[k] += gq * ld(dGTs + size_t(q) * nlmp + k * W);
      }
      for (int k = 0; k < nlv; ++k) st(ey + k * W, ay[k]);
    }
  }
  void layer0_dG(const Block& K, Work& w, int mu, const T* gA) const {
    dispatch_q(nbp / W, [&](auto nq) { this->template layer0_dG<decltype(nq)::value>(K, w, mu, gA); });
  }
  void layer0_bonds(Work& w, const T* blk, int nbs, int nbond) const {
    dispatch_q(nbp / W, [&](auto nq) {
      dispatch_q(nlmp / W, [&](auto nlv) {
        this->template layer0_bonds<decltype(nq)::value, decltype(nlv)::value>(w, blk, nbs, nbond);
      });
    });
  }

  // B1: the bond sums A of layer t (all blocks, rows of Cp) times f = a0 / sigma (1 + m / (sigma kappa)^2)^(-1/2),
  // m = mean of A^2; returns f
  T gate_apply(int t, T* A) const {
    double m = 0.0;
    for (size_t k = 0; k < lay[t].asz; ++k) m += double(A[k]) * double(A[k]);
    const T f = T(gate_c[t] / std::sqrt(1.0 + m / gate_n[t] / gate_s2[t]));
    for (size_t k = 0; k < lay[t].asz; ++k) A[k] *= f;
    return f;
  }
  // its backward: g = dE/dA' (A' = f A the gated sums, Ag) becomes dE/dA = f (g - h A'),
  // h = (g . A' + f gf) / (a0^2 kappa^2 N), gf = dE/df at fixed A
  void gate_back(int t, T f, const T* Ag, T* g, double gf) const {
    double dot = 0.0;
    for (size_t k = 0; k < lay[t].asz; ++k) dot += double(g[k]) * double(Ag[k]);
    const T h = T((dot + double(f) * gf) / (gate_k2 * gate_n[t]));
    for (size_t k = 0; k < lay[t].asz; ++k) g[k] = f * (g[k] - h * Ag[k]);
  }

  // without back-flow, j's message on a bond of atom i loses the reverse bond's order-1 part:
  // Ie[r] = I_j[r] - f_j (-1)^l Y_r(u) Q_l, Q_l = sum_q g_q bf[(mu_j, mu_i)][l][q][:], f_j: j's layer 0 gate factor
  template <int NV_> void backflow_Q(const T* tab, const T* blk, int nbs, int b, T* Q) const {
    const Block& K = lay[1].B[0];
    constexpr int NM = NV_ ? NV_ : MAXNV;
    const int nv = NV_ ? NV_ : K.NV, Cp = K.Cp, rg = R_G();
    for (int l = 0; l < nbf; ++l) {
      V acc[NM];
      for (int k = 0; k < nv; ++k) acc[k] = V{};
      for (int q = 0; q < nb; ++q) {
        const V s = splat(blk[size_t(rg + q) * nbs + b]);
        for (int k = 0; k < nv; ++k) acc[k] += s * ld(tab + (size_t(l) * nb + q) * Cp + k * W);
      }
      for (int k = 0; k < nv; ++k) st(Q + size_t(l) * Cp + k * W, acc[k]);
    }
  }
  template <int NV_>
  const T* backflow_fwd(Work& w, const T* tab, const T* blk, int nbs, int b, const T* Ij, T fj) const {
    const Block& K = lay[1].B[0];
    const int nv = NV_ ? NV_ : K.NV, Cp = K.Cp, ry = R_Y();
    backflow_Q<NV_>(tab, blk, nbs, b, w.Q.data());
    const T* Q = w.Q.data();
    T* Ie = w.Ie.data();
    for (int r = 0; r < nbf * nbf; ++r) {
      const T y = blk[size_t(ry + r) * nbs + b];
      const V s = splat(fj * ((deg[r] & 1) ? y : -y));
      const T* q = Q + size_t(deg[r]) * Cp;
      for (int k = 0; k < nv; ++k) st(Ie + size_t(r) * Cp + k * W, ld(Ij + size_t(r) * Cp + k * W) + s * ld(q + k * W));
    }
    return Ie;
  }

  static void prefetch(const T* p, size_t n) {
    for (size_t k = 0; k < n; k += 64 / sizeof(T)) __builtin_prefetch(p + k);
  }

  // A[o][:] += sum over bonds and terms of cg Y[lm] R[deg lm][:] I_j[r][:]
  template <int NV_, int NL_>
  void density(const Layer& Ly, Work& w, const Input& in, int i, const T* blk, int nbs, int nbond,
               const int32_t* bonds, const T* Ib, T* A) const {
    const Block& K = Ly.B[0];
    constexpr int NM = NV_ ? NV_ : MAXNV, LM = NL_ ? NL_ : MAXNL;
    const int nv = NV_ ? NV_ : K.NV, nl = NL_ ? NL_ : K.nl, Cp = K.Cp, rg = R_G(), ry = R_Y();
    std::memset(A, 0, sizeof(T) * size_t(K.nbase) * Cp);
    const size_t isz = size_t(Ly.nI) * Cp;
    const size_t nt = Ly.yt.size();
    for (int b = 0; b < nbond; ++b) {
      const int j = in.cj[bonds[b]];
      const T* Ij = Ib + size_t(j) * isz;
      if (b + 1 < nbond) prefetch(Ib + size_t(in.cj[bonds[b + 1]]) * isz, isz);
      const T* rc = ptab(in.mu[i], in.mu[j]) + K.pC;
      V R[LM][NM];
      for (int l = 0; l < nl; ++l)
        for (int k = 0; k < nv; ++k) R[l][k] = V{};
      for (int q = 0; q < nb; ++q) {
        const V s = splat(blk[size_t(rg + q) * nbs + b]);
        for (int l = 0; l < nl; ++l)
          for (int k = 0; k < nv; ++k) R[l][k] += s * ld(rc + (size_t(l) * nb + q) * Cp + k * W);
      }
      if (Ly.bf) Ij = backflow_fwd<NV_>(w, ptab(in.mu[j], in.mu[i]) + bC, blk, nbs, b, Ij, gate ? fgate[j] : T(1));
      size_t t = 0;
      while (t < nt) {
        const int o = Ly.yt[t].o;
        V acc[NM];
        for (int k = 0; k < nv; ++k) acc[k] = V{};
        for (; t < nt && Ly.yt[t].o == o; ++t) {
          const YTerm& y = Ly.yt[t];
          const V sv = splat(T(y.cg * blk[size_t(ry + y.lm) * nbs + b]));
          const T* ij = Ij + size_t(y.r) * Cp;
          auto term = [&](int l) {
            for (int k = 0; k < nv; ++k) acc[k] += sv * (R[l][k] * ld(ij + k * W));
          };
          if constexpr (NL_ > 0)
            select<NL_>(y.deg, term);
          else
            term(y.deg);
        }
        T* Ao = A + size_t(o) * Cp;
        for (int k = 0; k < nv; ++k) st(Ao + k * W, ld(Ao + k * W) + acc[k]);
      }
    }
  }
  void density(const Layer& Ly, Work& w, const Input& in, int i, const T* blk, int nbs, int nbond,
               const int32_t* bonds, const T* Ib, T* A) const {
    dispatch(Ly.B[0].NV, [&](auto nv) {
      dispatch_l(Ly.B[0].nl, [&](auto nl) {
        this->template density<decltype(nv)::value, decltype(nl)::value>(Ly, w, in, i, blk, nbs, nbond, bonds,
                                                                           Ib, A);
      });
    });
  }
  // the bonds of atom k in layer Ln (= t + 1): (a) the gradient of k's message, gathered from
  // the neighbours' gA through the reverse bonds, Y(-u) = (-1)^l Y(u) (with pair weights the
  // reverse pair's R); (b) dE/dg, dE/dY of each bond from k's own gA and the neighbour's message
  template <int NV_, int NL_>
  void back_bonds(const Layer& Ln, Work& w, const Input& in, int kk, const T* blk, int nbs, int nbond,
                  const int32_t* bonds, const T* gAk, const T* Ib, const T* gAb, T* gI) const {
    const Block& K = Ln.B[0];
    constexpr int NM = NV_ ? NV_ : MAXNV, LM = NL_ ? NL_ : MAXNL;
    const int nv = NV_ ? NV_ : K.NV, nl = NL_ ? NL_ : K.nl, Cp = K.Cp;
    const int rg = R_G(), ry = R_Y();
    constexpr int NY = NL_ ? NL_ * NL_ : (LMAX_SH + 1) * (LMAX_SH + 1);
    const int nlm = NL_ ? NL_ * NL_ : n_lm_feat;
    const size_t isz = size_t(Ln.nI) * Cp, asz = size_t(K.nbase) * Cp;
    const int mk = in.mu[kk];
    T* dRs = w.dR.data();
    for (int b = 0; b < nbond; ++b) {
      const int j = in.cj[bonds[b]];
      const T* Ij = Ib + size_t(j) * isz;
      const T* gAj = gAb + size_t(j) * asz;
      if (b + 1 < nbond) {
        const size_t jn = in.cj[bonds[b + 1]];
        prefetch(Ib + jn * isz, isz);
        prefetch(gAb + jn * asz, asz);
      }
      V R[LM][NM], dR[LM][NM], Rr[LM][NM];
      for (int l = 0; l < nl; ++l)
        for (int k = 0; k < nv; ++k) R[l][k] = dR[l][k] = V{};
      const T* rc = ptab(mk, in.mu[j]) + K.pC;
      for (int q = 0; q < nb; ++q) {
        const V s = splat(blk[size_t(rg + q) * nbs + b]);
        for (int l = 0; l < nl; ++l)
          for (int k = 0; k < nv; ++k) R[l][k] += s * ld(rc + (size_t(l) * nb + q) * Cp + k * W);
      }
      const T* rr = ptab(in.mu[j], mk) + K.pC;
      for (int l = 0; l < nl; ++l)
        for (int k = 0; k < nv; ++k) Rr[l][k] = V{};
      for (int q = 0; q < nb; ++q) {
        const V s = splat(blk[size_t(rg + q) * nbs + b]);
        for (int l = 0; l < nl; ++l)
          for (int k = 0; k < nv; ++k) Rr[l][k] += s * ld(rr + (size_t(l) * nb + q) * Cp + k * W);
      }
      T* ey = w.dEdY.data() + size_t(b) * nlmp;
      V dYv[NY];
      for (int lm = 0; lm < nlm; ++lm) dYv[lm] = V{};
      // lm (and so l) known at compile time when nl is: R, dR and dYv stay in registers
      auto term = [&](const YTerm& y, auto lmc) {
        constexpr int LMc = decltype(lmc)::value;
        const int lm = LMc >= 0 ? LMc : y.lm, l = LMc >= 0 ? isqrt(LMc) : y.deg;
        const T sY = y.cg * blk[size_t(ry + lm) * nbs + b];
        const V sa = splat((l & 1) ? T(-sY) : sY), sb = splat(sY);
        T* gi = gI + size_t(y.r) * Cp;
        const T* gaj = gAj + size_t(y.o) * Cp;
        const T* gak = gAk + size_t(y.o) * Cp;
        const T* ij = Ij + size_t(y.r) * Cp;
        V yacc{};
        for (int k = 0; k < nv; ++k) {
          const V rk = R[l][k];
          st(gi + k * W, ld(gi + k * W) + sa * (Rr[l][k] * ld(gaj + k * W)));
          const V tq = ld(gak + k * W) * ld(ij + k * W);
          dR[l][k] += sb * tq;
          yacc += tq * rk;
        }
        dYv[lm] += splat(y.cg) * yacc;
      };
      for (const YTerm& y : Ln.yt) {
        if constexpr (NL_ > 0)
          select<NL_ * NL_>(y.lm, [&](auto lmc) { term(y, lmc); });
        else
          term(y, std::integral_constant<int, -1>());
      }
      for (int lm = 0; lm < nlm; ++lm) ey[lm] += hsum(dYv[lm]);
      for (int l = 0; l < nl; ++l)
        for (int k = 0; k < nv; ++k) st(dRs + (size_t(b) * nl + l) * Cp + k * W, dR[l][k]);
    }
    dispatch_q(nbp / W, [&](auto nq) { this->template radial_back<decltype(nq)::value>(Ln, w, in, kk, nbond, bonds); });
  }
  // the same without back-flow: (c) the bond's gradient w.r.t. its message Ie, into dE/dg and dE/dY.
  // Terms by message row keep its gradients in registers; channels in chunks of CW vectors (R, dR too).
  template <int NV_, int NL_>
  void back_bonds_bf(const Layer& Ln, Work& w, const Input& in, int kk, const T* blk, int nbs, int nbond,
                     const int32_t* bonds, const T* gAk, const T* Ib, const T* gAb, T* gI) const {
    const Block& K = Ln.B[0];
    constexpr int LM = NL_ ? NL_ : MAXNL;
    const int nv = NV_ ? NV_ : K.NV, nl = NL_ ? NL_ : K.nl, Cp = K.Cp, C = K.C;
    const int rg = R_G(), ry = R_Y();
    constexpr int NY = NL_ ? NL_ * NL_ : (LMAX_SH + 1) * (LMAX_SH + 1);
    const int nlm = NL_ ? NL_ * NL_ : n_lm_feat;
    const size_t isz = size_t(Ln.nI) * Cp, asz = size_t(K.nbase) * Cp;
    const int mk = in.mu[kk];
    T* dRs = w.dR.data();
    T* dQ = w.dQ.data();
    const T* Q = w.Q.data();
    for (int b = 0; b < nbond; ++b) {
      const int j = in.cj[bonds[b]];
      const T* Ij = Ib + size_t(j) * isz;
      const T* gAj = gAb + size_t(j) * asz;
      if (b + 1 < nbond) {
        const size_t jn = in.cj[bonds[b + 1]];
        prefetch(Ib + jn * isz, isz);
        prefetch(gAb + jn * asz, asz);
      }
      backflow_Q<NV_>(ptab(in.mu[j], mk) + bC, blk, nbs, b, w.Q.data());
      const T fj = gate ? fgate[j] : T(1);
      if (gate) backflow_Q<NV_>(ptab(mk, in.mu[j]) + bC, blk, nbs, b, w.Qr.data());     // of the reverse bond
      double gfb = 0.0;
      for (int k = 0; k < nbf * nv; ++k) st(dQ + k * W, V{});
      T* ey = w.dEdY.data() + size_t(b) * nlmp;
      V dYv[NY];
      for (int lm = 0; lm < nlm; ++lm) dYv[lm] = V{};
      const T* rc = ptab(mk, in.mu[j]) + K.pC;
      const T* rr = ptab(in.mu[j], mk) + K.pC;
      // RR: the reverse bond has its own R (pair weights, two species)
      // vectors k0 .. k0 + cn of the channels (arrays of CW)
      auto chunk = [&](auto cw, auto rr_, int k0, int cn) {
        constexpr int CW = decltype(cw)::value;
        constexpr bool RR = decltype(rr_)::value;
        V R[LM][CW], dR[LM][CW], Rr[RR ? LM : 1][RR ? CW : 1];
        for (int l = 0; l < nl; ++l)
          for (int k = 0; k < cn; ++k) R[l][k] = dR[l][k] = V{};
        for (int q = 0; q < nb; ++q) {
          const V s = splat(blk[size_t(rg + q) * nbs + b]);
          for (int l = 0; l < nl; ++l)
            for (int k = 0; k < cn; ++k) R[l][k] += s * ld(rc + (size_t(l) * nb + q) * Cp + (k0 + k) * W);
        }
        if constexpr (RR) {
          for (int l = 0; l < nl; ++l)
            for (int k = 0; k < cn; ++k) Rr[l][k] = V{};
          for (int q = 0; q < nb; ++q) {
            const V s = splat(blk[size_t(rg + q) * nbs + b]);
            for (int l = 0; l < nl; ++l)
              for (int k = 0; k < cn; ++k) Rr[l][k] += s * ld(rr + (size_t(l) * nb + q) * Cp + (k0 + k) * W);
          }
        }
        for (int r = 0; r < nbf * nbf; ++r) {
          const int lr = deg[r];
          const T yr = blk[size_t(ry + r) * nbs + b];
          const V sy = splat(fj * ((lr & 1) ? yr : -yr));
          T* gi = gI + size_t(r) * Cp + k0 * W;
          const T* qr = Q + size_t(lr) * Cp + k0 * W;
          V ga[CW], ge[CW], ie[CW];
          for (int k = 0; k < cn; ++k) {
            ga[k] = gate ? V{} : ld(gi + k * W);
            ge[k] = V{};
            ie[k] = ld(Ij + size_t(r) * Cp + (k0 + k) * W) + sy * ld(qr + k * W);
          }
          auto term = [&](const YTerm& y, auto lmc) {
            constexpr int LMc = decltype(lmc)::value;
            const int lm = LMc >= 0 ? LMc : y.lm, l = LMc >= 0 ? isqrt(LMc) : y.deg;
            const T sY = y.cg * blk[size_t(ry + lm) * nbs + b];
            const V sa = splat((l & 1) ? T(-sY) : sY), sb = splat(sY);
            const T* gaj = gAj + size_t(y.o) * Cp + k0 * W;
            const T* gak = gAk + size_t(y.o) * Cp + k0 * W;
            V yacc{};
            for (int k = 0; k < cn; ++k) {
              const V rk = R[l][k], gk = ld(gak + k * W);
              if constexpr (RR)
                ga[k] += sa * (Rr[l][k] * ld(gaj + k * W));
              else
                ga[k] += sa * (rk * ld(gaj + k * W));
              const V tq = gk * ie[k];
              dR[l][k] += sb * tq;
              yacc += tq * rk;
              ge[k] += sb * (rk * gk);
            }
            dYv[lm] += splat(y.cg) * yacc;
          };
          for (int t = Ln.yr_off[r]; t < Ln.yr_off[r + 1]; ++t) {
            const YTerm& y = Ln.ytr[t];
            if constexpr (NL_ > 0)
              select<NL_ * NL_>(y.lm, [&](auto lmc) { term(y, lmc); });
            else
              term(y, std::integral_constant<int, -1>());
          }
          // dIe/dD = -f_j for D = (-1)^l Y Q
          const T sg = (lr & 1) ? T(1) : T(-1);
          const V sq = splat(fj * (sg * yr));
          T* dq = dQ + size_t(lr) * Cp + k0 * W;
          V acc{};
          if (gate) {
            // ga is this bond's share of dE/dI_kk[r], taken by the reverse bond (centre j, sender kk) with
            // Ie = I_kk - f_kk Y(u) Q', Y(u) of this bond, Q' the reverse bond's back-flow
            const T* qq = w.Qr.data() + size_t(lr) * Cp + k0 * W;
            V accr{};
            for (int k = 0; k < cn; ++k) {
              accr += ga[k] * ld(qq + k * W);
              ga[k] += ld(gi + k * W);
            }
            gfb -= double(yr) * double(hsum_tree(accr));
          }
          for (int k = 0; k < cn; ++k) {
            st(gi + k * W, ga[k]);
            acc += ge[k] * ld(qr + k * W);
            st(dq + k * W, ld(dq + k * W) + sq * ge[k]);
          }
          ey[r] += (fj * sg) * hsum_tree(acc);
        }
        for (int l = 0; l < nl; ++l)
          for (int k = 0; k < cn; ++k) st(dRs + (size_t(b) * nl + l) * Cp + (k0 + k) * W, dR[l][k]);
      };
      // up to three vectors in one pass, more in pairs: a term's scalar work repeats per chunk,
      // and wide chunks spill
      auto chunks = [&](auto rr_) {
        using Two = std::integral_constant<int, 2>;
        if constexpr (NV_ > 0 && NV_ <= 3) {
          chunk(std::integral_constant<int, NV_>(), rr_, 0, NV_);
        } else {
          int k0 = 0;
          for (; k0 + 2 <= nv; k0 += 2) chunk(Two(), rr_, k0, 2);
          if (k0 < nv) chunk(Two(), rr_, k0, 1);
        }
      };
      if (in.mu[j] != mk)
        chunks(std::true_type());
      else
        chunks(std::false_type());
      w.gf0 += gfb;
      const T* tabQ = ptab(in.mu[j], mk) + bQ;
      T* eg = w.dEdg.data() + size_t(b) * nbp;
      dispatch_q(nbp / W, [&](auto nqc) {
        constexpr int NQ_ = decltype(nqc)::value, QM = NQ_ ? NQ_ : MAXQ;
        const int nq = NQ_ ? NQ_ : nbp / W;
        V q0[QM], q1[QM];
        for (int k = 0; k < nq; ++k) {
          q0[k] = ld(eg + k * W);
          q1[k] = V{};
        }
        for (int l = 0; l < nbf; ++l) {
          const T* wq = tabQ + size_t(l) * C * nbp;
          const T* d = dQ + size_t(l) * Cp;
          int c = 0;
          for (; c + 1 < C; c += 2) {
            const V s0 = splat(d[c]), s1 = splat(d[c + 1]);
            for (int k = 0; k < nq; ++k) {
              q0[k] += s0 * ld(wq + size_t(c) * nbp + k * W);
              q1[k] += s1 * ld(wq + size_t(c + 1) * nbp + k * W);
            }
          }
          if (c < C) {
            const V s0 = splat(d[c]);
            for (int k = 0; k < nq; ++k) q0[k] += s0 * ld(wq + size_t(c) * nbp + k * W);
          }
        }
        for (int k = 0; k < nq; ++k) st(eg + k * W, q0[k] + q1[k]);
      });
      for (int lm = 0; lm < nlm; ++lm) ey[lm] += hsum_tree(dYv[lm]);
    }
    dispatch_q(nbp / W, [&](auto nq) { this->template radial_back<decltype(nq)::value>(Ln, w, in, kk, nbond, bonds); });
  }
  // dE/dg of every bond += sum_{l, c} dR[l][c] tab[pQ + (l C + c) nbp + :] (dR of the bonds in w.dR)
  template <int NQ_> void radial_back(const Layer& Ln, Work& w, const Input& in, int kk, int nbond,
                                      const int32_t* bonds) const {
    const Block& K = Ln.B[0];
    constexpr int QM = NQ_ ? NQ_ : MAXQ;
    const int nq = NQ_ ? NQ_ : nbp / W, nl = K.nl, C = K.C, Cp = K.Cp;
    for (int b = 0; b < nbond; ++b) {
      const T* dRs = w.dR.data() + size_t(b) * nl * Cp;
      const T* Q0 = ptab(in.mu[kk], in.mu[in.cj[bonds[b]]]) + K.pQ;
      T* eg = w.dEdg.data() + size_t(b) * nbp;
      V q0[QM], q1[QM];
      for (int k = 0; k < nq; ++k) {
        q0[k] = ld(eg + k * W);
        q1[k] = V{};
      }
      for (int l = 0; l < nl; ++l) {
        const T* wq = Q0 + size_t(l) * C * nbp;
        const T* d = dRs + size_t(l) * Cp;
        int c = 0;
        for (; c + 1 < C; c += 2) {
          const V s0 = splat(d[c]), s1 = splat(d[c + 1]);
          for (int k = 0; k < nq; ++k) {
            q0[k] += s0 * ld(wq + size_t(c) * nbp + k * W);
            q1[k] += s1 * ld(wq + size_t(c + 1) * nbp + k * W);
          }
        }
        if (c < C) {
          const V s0 = splat(d[c]);
          for (int k = 0; k < nq; ++k) q0[k] += s0 * ld(wq + size_t(c) * nbp + k * W);
        }
      }
      for (int k = 0; k < nq; ++k) st(eg + k * W, q0[k] + q1[k]);
    }
  }
  void back_bonds(const Layer& Ln, Work& w, const Input& in, int kk, const T* blk, int nbs, int nbond,
                  const int32_t* bonds, const T* gAk, const T* Ib, const T* gAb, T* gI) const {
    dispatch(Ln.B[0].NV, [&](auto nv) {
      dispatch_l(Ln.B[0].nl, [&](auto nl) {
        constexpr int NV_ = decltype(nv)::value, NL_ = decltype(nl)::value;
        if (Ln.bf)
          this->template back_bonds_bf<NV_, NL_>(Ln, w, in, kk, blk, nbs, nbond, bonds, gAk, Ib, gAb, gI);
        else
          this->template back_bonds<NV_, NL_>(Ln, w, in, kk, blk, nbs, nbond, bonds, gAk, Ib, gAb, gI);
      });
    });
  }

  // C[:Mp][:Np] = A @ B: A by rows, B in panels of NR columns
  static void gemm(int Mp, int K, int Np, const T* A, int lda, const T* Bp, T* Cm, int ldc) {
    for (int p = 0; p < Np / NR; ++p) {
      const T* B = Bp + size_t(p) * K * NR;
      for (int i = 0; i < Mp; i += MR) kernel(K, A + size_t(i) * lda, lda, B, Cm + size_t(i) * ldc + size_t(p) * NR, ldc);
    }
  }
  static inline void kernel(int kc, const T* __restrict A, int lda, const T* __restrict B, T* __restrict Cm, int ldc) {
    V c00{}, c01{}, c10{}, c11{}, c20{}, c21{}, c30{}, c31{}, c40{}, c41{}, c50{}, c51{};
    const T *a0 = A, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda, *a4 = a3 + lda, *a5 = a4 + lda;
    for (int k = 0; k < kc; ++k) {
      const V b0 = ld(B), b1 = ld(B + W);
      B += NR;
      T x;
      x = a0[k]; c00 += x * b0; c01 += x * b1;
      x = a1[k]; c10 += x * b0; c11 += x * b1;
      x = a2[k]; c20 += x * b0; c21 += x * b1;
      x = a3[k]; c30 += x * b0; c31 += x * b1;
      x = a4[k]; c40 += x * b0; c41 += x * b1;
      x = a5[k]; c50 += x * b0; c51 += x * b1;
    }
    T* c = Cm;
    st(c, c00); st(c + W, c01); c += ldc;
    st(c, c10); st(c + W, c11); c += ldc;
    st(c, c20); st(c + W, c21); c += ldc;
    st(c, c30); st(c + W, c31); c += ldc;
    st(c, c40); st(c + W, c41); c += ldc;
    st(c, c50); st(c + W, c51);
  }

  // site energies of rows 0..m-1 of x (block of Mp rows, species mu[]), and gx = dE/dx
  // with the core switch: cw the switch of each row; hs the readout's sum over the head, before scale
  void readout(Work& w, int m, int Mp, const int32_t* mu, double* e, const double* cw, double* hs) const {
    const int H = int(mlp.size());
    for (int h = 0; h < H; ++h) {
      const Dense& D = mlp[h];
      const T* in = h ? w.a[h - 1].data() : w.x.data();
      const int lda = h ? mlp[h - 1].nout_p : F_ld;
      T* z = w.z[h].data();
      gemm(Mp, D.nin, D.nout_p, in, lda, D.Wf.data(), z, D.nout_p);
      T* a = w.a[h].data();
      T* s = w.s[h].data();
      for (int i = 0; i < Mp; ++i) {
        const T* b = D.bias.data() + (h == 0 ? size_t(i < m ? mu[i] : 0) * D.nout_p : 0);
        T* zi = z + size_t(i) * D.nout_p;
        for (int n = 0; n < D.nout_p; n += W) {
          const V zz = ld(zi + n) + ld(b + n);
          const V sg = splat(T(1)) / (splat(T(1)) + vexp(-zz));
          st(zi + n, zz);
          st(s + size_t(i) * D.nout_p + n, sg);
          st(a + size_t(i) * D.nout_p + n, zz * sg);
        }
      }
    }
    // energies and the gradient at the last layer's pre-activation
    const Dense& DL = mlp.back();
    for (int i = 0; i < Mp; ++i) {
      const T* hd = head.data() + size_t(i < m ? mu[i] : 0) * DL.nout_p;
      const T* a = w.a[H - 1].data() + size_t(i) * DL.nout_p;
      const T* z = w.z[H - 1].data() + size_t(i) * DL.nout_p;
      const T* s = w.s[H - 1].data() + size_t(i) * DL.nout_p;
      T* g = w.g[H - 1].data() + size_t(i) * DL.nout_p;
      const T sc = T(cw && i < m ? scale * cw[i] : scale);
      V acc{};
      for (int n = 0; n < DL.nout_p; n += W) {
        const V hv = ld(hd + n), sg = ld(s + n), zz = ld(z + n);
        acc += hv * ld(a + n);
        st(g + n, (sc * hv) * (sg * (splat(T(1)) + zz * (splat(T(1)) - sg))));
      }
      if (i < m) {
        e[i] = double(hsum(acc)) * scale + e0[mu[i]];
        if (hs) hs[i] = double(hsum(acc));
      }
    }
    for (int h = H - 1; h >= 0; --h) {
      const Dense& D = mlp[h];
      T* gin = h ? w.g[h - 1].data() : w.gx.data();
      const int ldg = h ? mlp[h - 1].nout_p : mlp[0].nin_p;
      gemm(Mp, D.nout, D.nin_p, w.g[h].data(), D.nout_p, D.Wb.data(), gin, ldg);
      if (h) {                                       // through the previous activation
        const Dense& Dp = mlp[h - 1];
        for (int i = 0; i < Mp; ++i) {
          T* gi = gin + size_t(i) * Dp.nout_p;
          const T* z = w.z[h - 1].data() + size_t(i) * Dp.nout_p;
          const T* s = w.s[h - 1].data() + size_t(i) * Dp.nout_p;
          for (int n = 0; n < Dp.nout_p; n += W) {
            const V sg = ld(s + n), zz = ld(z + n);
            st(gi + n, ld(gi + n) * (sg * (splat(T(1)) + zz * (splat(T(1)) - sg))));
          }
        }
      }
    }
  }

  // out[:ld] = sum_f sum_c A[rows[f * step]][c] Wm[f][c][:]
  template <int NV_> static void project(const Block& K, const T* A, const int* rows, int step, int nf, const T* Wm,
                                         int ld_, T* out) {
    constexpr int NM = NV_ ? NV_ : MAXNV;
    const int nv = NV_ ? NV_ : ld_ / W;
    V acc[NM];
    for (int k = 0; k < nv; ++k) acc[k] = V{};
    for (int f = 0; f < nf; ++f) {
      const T* a = A + size_t(rows[size_t(f) * step]) * K.Cp;
      const T* wf = Wm + size_t(f) * K.C * ld_;
      for (int c = 0; c < K.C; ++c) {
        const V s = splat(a[c]);
        for (int k = 0; k < nv; ++k) acc[k] += s * ld(wf + size_t(c) * ld_ + k * W);
      }
    }
    for (int k = 0; k < nv; ++k) st(out + k * W, acc[k]);
  }
  // the last layer's projections: per degree Pg (2l+1 rows of rp), the bispectrum's base rows in Pb
  void gram_project(const Block& K, const T* A, T* Pg, T* Pb) const {
    const Gram& g = gram;
    const Block& Bk = g.bis;
    for (int l = 0; l <= lmax_feat; ++l) {
      const int nm = 2 * l + 1, nf = int(g.rows[l].size()) / nm;
      for (int mm = 0; mm < nm; ++mm) {
        const int* rows = g.rows[l].data() + mm;
        T* p = Pg + g.poff[l] + size_t(mm) * g.rp;
        T* q = Pb + size_t(l * l + mm) * Bk.Cp;
        dispatch(g.rp / W, [&](auto nv) { project<decltype(nv)::value>(K, A, rows, nm, nf, g.G[l].data(), g.rp, p); });
        dispatch(Bk.NV, [&](auto nv) { project<decltype(nv)::value>(K, A, rows, nm, nf, g.Bm[l].data(), Bk.Cp, q); });
      }
    }
  }
  // the last layer's features from its A: its scalars, the Gram matrices, the bispectrum
  void gram_forward(const Block& K, Work& w, const T* A, T* x) const {
    const Gram& g = gram;
    T* Pg = w.Pg.data();
    T* Pb = w.Pb.data();
    copy_features(K, A, x);
    gram_project(K, A, Pg, Pb);
    for (int l = 0; l <= lmax_feat; ++l) {
      const T* P = Pg + g.poff[l];
      T* out = x + g.foff + l * g.npair;
      for (int i = 0; i < g.r; ++i)
        for (int j = i; j < g.r; j += W) {
          V acc{};
          for (int mm = 0; mm < 2 * l + 1; ++mm) acc += splat(P[size_t(mm) * g.rp + i]) * ld(P + size_t(mm) * g.rp + j);
          const int n = std::min(W, g.r - j);
          if (n == W)
            st(out, acc);
          else
            for (int e = 0; e < n; ++e) out[e] = acc[e];
          out += n;
        }
    }
    products_forward(g.bis, Pb);
    copy_features(g.bis, Pb, x);
  }
  // gx (the features' gradient) -> gA of the last layer (nbase rows)
  void gram_back(const Block& K, Work& w, const T* A, const T* gx, T* gA) const {
    const Gram& g = gram;
    const Block& Bk = g.bis;
    T* Pg = w.Pg.data();
    T* dPg = w.dPg.data();
    T* Pb = w.Pb.data();
    T* gPb = w.gPb.data();
    T* Gs = w.Gs.data();
    gram_project(K, A, Pg, Pb);
    products_forward(Bk, Pb);
    std::memset(gPb, 0, sizeof(T) * size_t(Bk.nrows) * Bk.Cp);
    add_features(Bk, gx, gPb);
    products_backward(Bk, Pb, gPb);
    const int r = g.r, nvr = g.rp / W;
    for (int l = 0; l <= lmax_feat; ++l) {
      const T* gg = gx + g.foff + l * g.npair;
      for (int i = 0; i < r; ++i) {
        Gs[size_t(i) * g.rp + i] = 2 * *gg++;
        for (int j = i + 1; j < r; ++j, ++gg) {
          Gs[size_t(i) * g.rp + j] = *gg;
          Gs[size_t(j) * g.rp + i] = *gg;
        }
      }
      dispatch(nvr, [&](auto nvc) {
        constexpr int NV_ = decltype(nvc)::value, NM = NV_ ? NV_ : MAXNV;
        const int nv = NV_ ? NV_ : nvr;
        for (int mm = 0; mm < 2 * l + 1; ++mm) {
          const T* P = Pg + g.poff[l] + size_t(mm) * g.rp;
          T* dP = dPg + g.poff[l] + size_t(mm) * g.rp;
          V acc[NM];
          for (int k = 0; k < nv; ++k) acc[k] = V{};
          for (int b = 0; b < r; ++b) {
            const V s = splat(P[b]);
            for (int k = 0; k < nv; ++k) acc[k] += s * ld(Gs + size_t(b) * g.rp + k * W);
          }
          for (int k = 0; k < nv; ++k) st(dP + k * W, acc[k]);
        }
      });
    }
    std::memset(gA, 0, sizeof(T) * size_t(K.nbase) * K.Cp);
    add_features(K, gx, gA);
    dispatch(K.NV, [&](auto nvc) {
      constexpr int NV_ = decltype(nvc)::value, NM = NV_ ? NV_ : MAXNV;
      const int nv = NV_ ? NV_ : K.NV;
      for (int l = 0; l <= lmax_feat; ++l) {
        const int nm = 2 * l + 1, nf = int(g.rows[l].size()) / nm;
        for (int mm = 0; mm < nm; ++mm) {
          const T* dP = dPg + g.poff[l] + size_t(mm) * g.rp;
          const T* dq = gPb + size_t(l * l + mm) * Bk.Cp;
          for (int f = 0; f < nf; ++f) {
            T* d = gA + size_t(g.rows[l][size_t(f) * nm + mm]) * K.Cp;
            const T* GT = g.GT[l].data() + size_t(f) * r * K.Cp;
            const T* BT = g.BT[l].data() + size_t(f) * g.rb * K.Cp;
            V acc[NM];
            for (int k = 0; k < nv; ++k) acc[k] = ld(d + k * W);
            for (int c = 0; c < r; ++c) {
              const V s = splat(dP[c]);
              for (int k = 0; k < nv; ++k) acc[k] += s * ld(GT + size_t(c) * K.Cp + k * W);
            }
            for (int c = 0; c < g.rb; ++c) {
              const V s = splat(dq[c]);
              for (int k = 0; k < nv; ++k) acc[k] += s * ld(BT + size_t(c) * K.Cp + k * W);
            }
            for (int k = 0; k < nv; ++k) st(d + k * W, acc[k]);
          }
        }
      }
    });
  }

  void prepare(int n, int64_t n_cand, int nthreads) {
    Abuf.resize(L);
    Ibuf.resize(L);
    gAbuf.resize(L);
    gfeat.resize(L);
    for (int t = 0; t < L; ++t) {
      const Layer& Ly = lay[t];
      if (t < L - 1) Abuf[t].reserve(size_t(n) * Ly.asz);
      if (t >= 1) {
        Ibuf[t].reserve(size_t(n) * Ly.nI * Ly.B[0].Cp);
        gAbuf[t].reserve(size_t(n) * Ly.asz);
      }
      if (t < L - 1) gfeat[t].reserve(size_t(n) * Ly.gsz);
    }
    nact.reserve(size_t(n));
    act.reserve(size_t(std::max<int64_t>(n_cand, 1)));
    dvec.reserve(size_t(3 * std::max<int64_t>(n_cand, 1)));
    eat.reserve(size_t(n));
    if (gate) fgate.reserve(size_t(L) * n);
    if (core) {
      core_c.reserve(size_t(n));
      core_dc.reserve(size_t(n));
    }
    goff.reserve(size_t(n) + 1);
    if (int(work.size()) < nthreads) work.resize(nthreads);
  }
  void prepare_work(Work& w, int maxdeg) const {
    const int Mp = MB, deg_ = int(round_up(std::max(maxdeg, 1), W));
    w.x.reserve(size_t(Mp) * F_ld);
    w.gx.reserve(size_t(Mp) * mlp[0].nin_p);
    const int H = int(mlp.size());
    if (int(w.z.size()) < H) {
      w.z.resize(H);
      w.a.resize(H);
      w.s.resize(H);
      w.g.resize(H);
    }
    for (int h = 0; h < H; ++h) {
      const size_t sz = size_t(Mp) * mlp[h].nout_p;
      w.z[h].reserve(sz);
      w.a[h].reserve(sz);
      w.s[h].reserve(sz);
      w.g[h].reserve(sz);
    }
    size_t rows = 0, asz = 0, ni = 0, rl = 0;
    for (const Layer& Ly : lay) {
      for (const Block& K : Ly.B) {
        rows = std::max(rows, size_t(K.nrows) * K.Cp);
        rl = std::max(rl, size_t(K.nl) * K.Cp);
      }
      asz = std::max(asz, Ly.asz);
      ni = std::max(ni, size_t(std::max(Ly.nI, 1)) * Ly.B[0].Cp);
    }
    w.P.reserve(rows);
    w.gP.reserve(rows);
    w.gI.reserve(ni);
    w.dR.reserve(size_t(deg_) * rl);
    w.blockA.reserve(size_t(MB) * asz);
    if (gate) w.gA0.reserve(lay[0].asz);
    if (squash) w.dx.reserve(size_t(Mp) * F_ld);
    w.dEdg.reserve(size_t(deg_) * nbp);
    w.dEdY.reserve(size_t(deg_) * nlmp);
    w.ga.reserve(size_t(deg_) * nbp);
    w.dgeo.reserve(size_t(deg_) * (nb + 3 * n_lm));
    w.G.reserve(size_t(deg_) * n_lm * nbp);
    w.dG.reserve(size_t(deg_) * n_lm * nbp);
    w.dGT.reserve(size_t(deg_) * nb * nlmp);
    w.Ie.reserve(ni);
    w.Q.reserve(size_t(nbf) * lay[1].B[0].Cp);
    w.Qr.reserve(size_t(nbf) * lay[1].B[0].Cp);
    w.dQ.reserve(size_t(nbf) * lay[1].B[0].Cp);
    if (!w.Pg.p) w.Pg.zeros(gram.psz + W);         // the triangle's vectors read past a row's end
    w.dPg.reserve(gram.psz);
    if (!w.Gs.p) w.Gs.zeros(size_t(gram.r) * gram.rp);
    w.Pb.reserve(size_t(gram.bis.nrows) * gram.bis.Cp);
    w.gPb.reserve(size_t(gram.bis.nrows) * gram.bis.Cp);
    if (int(w.bspec.size()) < deg_) w.bspec.resize(deg_);
    w.slist.reserve(deg_);
  }
  void copy_features(const Block& Ly, const T* P, T* x) const {
    for (size_t f = 0; f < Ly.frows.size(); ++f)
      std::memcpy(x + Ly.foff + f * Ly.C, P + size_t(Ly.frows[f]) * Ly.Cp, sizeof(T) * Ly.C);
  }
  void add_features(const Block& Ly, const T* gx, T* gP) const {
    for (size_t f = 0; f < Ly.frows.size(); ++f) {
      T* d = gP + size_t(Ly.frows[f]) * Ly.Cp;
      const T* s = gx + Ly.foff + f * Ly.C;
      for (int c = 0; c < Ly.C; ++c) d[c] += s[c];
    }
  }
  void zero_bond_grads(Work& w, int nbond) const {
    std::memset(w.dEdg.data(), 0, sizeof(T) * size_t(nbond) * nbp);
    std::memset(w.dEdY.data(), 0, sizeof(T) * size_t(nbond) * nlmp);
  }

  // BACK stage of layer t (t + 1's gradient known): atom k
  void back_atom(Work& w, const Input& in, int t, int k, int nbond, const int32_t* bonds) {
    const Layer& Ln = lay[t + 1];
    const Layer& Ly = lay[t];
    const Block& Kn = Ln.B[0];
    const T* blk = geo.data() + goff[k];
    const int nbs = int(round_up(nbond, W));
    if (t == 0) species_map(w, in, nbond, bonds);
    zero_bond_grads(w, nbond);
    T* gI = w.gI.data();
    std::memset(gI, 0, sizeof(T) * size_t(Ln.nI) * Kn.Cp);
    w.gf0 = 0.0;
    back_bonds(Ln, w, in, k, blk, nbs, nbond, bonds, gAbuf[t + 1].data() + size_t(k) * Ln.asz,
               Ibuf[t + 1].data(), gAbuf[t + 1].data(), gI);
    // gP of each block of layer t: readout features + the reduction; then its products
    T* gP = w.gP.data();
    T* P = w.P.data();
    for (size_t bi = 0; bi < Ly.B.size(); ++bi) {
      const Block& K = Ly.B[bi];
      std::memset(gP, 0, sizeof(T) * size_t(K.nrows) * K.Cp);
      const T* gf = gfeat[t].data() + size_t(k) * Ly.gsz + K.goff;
      for (size_t f = 0; f < K.frows.size(); ++f) {
        T* d = gP + size_t(K.frows[f]) * K.Cp;
        const T* s = gf + f * K.Cp;
        for (int q = 0; q < K.NV; ++q) st(d + q * W, ld(d + q * W) + ld(s + q * W));
      }
      reduce_bwd(t + 1, int(bi), gI, gP);
      std::memcpy(P, Abuf[t].data() + size_t(k) * Ly.asz + K.aoff, sizeof(T) * size_t(K.nbase) * K.Cp);
      products_forward(K, P);
      products_backward(K, P, gP);
      if (t >= 1)
        std::memcpy(gAbuf[t].data() + size_t(k) * Ly.asz, gP, sizeof(T) * size_t(K.nbase) * K.Cp);
      else if (gate)
        std::memcpy(w.gA0.data() + K.aoff, gP, sizeof(T) * size_t(K.nbase) * K.Cp);
      else
        layer0_dG(K, w, in.mu[k], gP);
    }
    if (gate) {
      T* gA = t >= 1 ? gAbuf[t].data() + size_t(k) * Ly.asz : w.gA0.data();
      gate_back(t, fgate[size_t(t) * in.n + k], Abuf[t].data() + size_t(k) * Ly.asz, gA, w.gf0);
      if (t == 0)
        for (const Block& K : Ly.B) layer0_dG(K, w, in.mu[k], gA + K.aoff);
    }
    if (t == 0) layer0_bonds(w, blk, nbs, nbond);
    finish_bonds(w, in, k, blk, nbs, nbond, bonds);
  }

  int compute(const Input& in, double* energy, double* e_atoms, double* forces, double* virial, int64_t* nbonds,
              int nthreads) {
    const int n = in.n;
    const int64_t n_cand = in.off[n];
    if (nthreads <= 0) {
#ifdef _OPENMP
      nthreads = omp_get_max_threads();
#else
      nthreads = 1;
#endif
    }
    nthreads = std::max(1, std::min(nthreads, (n + 7) / 8));     // at least 8 atoms per thread
    // readout blocks: a whole number of register blocks, enough blocks for every thread
    // (above 48 rows at least eight each, for the dynamic schedule)
    MB = int(std::min<int64_t>(mb_max, round_up((n + nthreads - 1) / nthreads, MR)));
    MB = std::max({std::min<int>(MB, 48), std::min<int>(MB, int(round_up(n / (8 * nthreads), MR))), MR});
    prepare(n, n_cand, nthreads);
    build_pairs(in);
    int maxdeg = 0;
    for (int i = 0; i < n; ++i) maxdeg = std::max<int>(maxdeg, int(in.off[i + 1] - in.off[i]));
    for (int t = 0; t < nthreads; ++t) prepare_work(work[t], maxdeg);
    const double rc2 = rcut * rcut;
    const int nblocks = (n + MB - 1) / MB;
    const int chunk = std::max(1, std::min(32, n / (4 * nthreads)));    // atoms per piece of the schedule
    double vir[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string err;
    double clk[6];
    auto now = []() {
#ifdef _OPENMP
      return omp_get_wtime();
#else
      return 0.0;
#endif
    };
    clk[0] = now();
    // FILTER: the candidates inside rcut, ZBL (nothing here can throw)
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int i = 0; i < n; ++i) {
      const int64_t o0 = in.off[i], o1 = in.off[i + 1];
      int32_t* bonds = act.data() + o0;
      int nbond = 0;
      double ez = 0.0;
      for (int64_t c = o0; c < o1; ++c) {
        const int j = in.cj[c];
        const double* sh = in.csh + 3 * c;
        const double vx = in.pos[3 * j] - in.pos[3 * i] + sh[0];
        const double vy = in.pos[3 * j + 1] - in.pos[3 * i + 1] + sh[1];
        const double vz = in.pos[3 * j + 2] - in.pos[3 * i + 2] + sh[2];
        const double r2 = vx * vx + vy * vy + vz * vz;
        if (!(r2 < rc2)) continue;
        if (!vc_rc2.empty() && !(r2 < vc_rc2[size_t(in.mu[i]) * S + in.mu[j]])) continue;
        bonds[nbond++] = int32_t(c);
        T* dv = dvec.data() + 3 * c;
        dv[0] = dv[1] = dv[2] = T(0);
        if (zbl) {
          const int mi = in.mu[i], mj = in.mu[j];
          const double rmax = std::min(zbl_factor * (zbl_rcov[mi] + zbl_rcov[mj]), rcut);
          const double r = std::max(std::sqrt(r2), 1e-8);
          if (r < rmax) {
            double e, de;
            repulsion(r, zbl_z[mi], zbl_z[mj], rmax, &e, &de);
            ez += 0.5 * e;
            dv[0] = T(0.5 * de * vx / r);
            dv[1] = T(0.5 * de * vy / r);
            dv[2] = T(0.5 * de * vz / r);
          }
        }
      }
      nact[i] = nbond;
      eat[i] = ez;
      if (core) {
        const double u = std::min(std::max(ez / zc, 0.0), 1.0), v = 1.0 - u;
        core_c[i] = 1.0 - u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
        core_dc[i] = -30.0 * u * u * v * v / zc;
      }
    }
    // every atom's geometry block (outside OpenMP: an allocation failure must reach the caller)
    goff[0] = 0;
    for (int i = 0; i < n; ++i) goff[i + 1] = goff[i] + int64_t(geo_block(int(round_up(nact[i], W))));
    geo.reserve(size_t(std::max<int64_t>(goff[n], 1)));
#pragma omp parallel num_threads(nthreads)
    {
#ifdef _OPENMP
      Work& w = work[omp_get_thread_num()];
      const bool t0 = omp_get_thread_num() == 0;
#else
      Work& w = work[0];
      const bool t0 = true;
#endif
      try {
        // PRE: geometry; layer 0 and the first message
#pragma omp for schedule(dynamic, chunk)
        for (int i = 0; i < n; ++i) {
          const int nbond = nact[i];
          const int32_t* bonds = act.data() + in.off[i];
          T* blk = geo.data() + goff[i];
          bond_block(in, i, nbond, bonds, blk);
          const Layer& L0 = lay[0];
          const Layer& L1 = lay[1];
          const int nbs = int(round_up(nbond, W));
          species_map(w, in, nbond, bonds);
          T* A = Abuf[0].data() + size_t(i) * L0.asz;
          layer0_density(w, blk, nbs, nbond, in.mu[i], A);
          if (gate) fgate[i] = gate_apply(0, A);
          T* I = Ibuf[1].data() + size_t(i) * L1.nI * L1.B[0].Cp;
          std::memset(I, 0, sizeof(T) * size_t(L1.nI) * L1.B[0].Cp);
          for (size_t bi = 0; bi < L0.B.size(); ++bi) {
            const Block& K = L0.B[bi];
            std::memcpy(w.P.data(), A + K.aoff, sizeof(T) * size_t(K.nbase) * K.Cp);
            products_forward(K, w.P.data());
            reduce_add(1, int(bi), w.P.data(), I);
          }
        }
        if (t0) clk[1] = now();
        // MID: layers 1 .. L-2
        for (int t = 1; t < L - 1; ++t) {
          const Layer& Ly = lay[t];
          const Layer& Ln = lay[t + 1];
          const Block& K = Ly.B[0];
#pragma omp for schedule(dynamic, chunk)
          for (int i = 0; i < n; ++i) {
            const int nbond = nact[i];
            const int32_t* bonds = act.data() + in.off[i];
            T* A = Abuf[t].data() + size_t(i) * Ly.asz;
            density(Ly, w, in, i, geo.data() + goff[i], int(round_up(nbond, W)), nbond, bonds, Ibuf[t].data(), A);
            if (gate) fgate[size_t(t) * n + i] = gate_apply(t, A);
            std::memcpy(w.P.data(), A, sizeof(T) * size_t(K.nbase) * K.Cp);
            products_forward(K, w.P.data());
            T* I = Ibuf[t + 1].data() + size_t(i) * Ln.nI * Ln.B[0].Cp;
            std::memset(I, 0, sizeof(T) * size_t(Ln.nI) * Ln.B[0].Cp);
            reduce_add(t + 1, 0, w.P.data(), I);
          }
        }
        if (t0) clk[2] = now();
        // LAST: the last layer, the features, the readout, the last layer's gradient
        const int tl = L - 1;
        const Layer& LL = lay[tl];
#pragma omp for schedule(dynamic, 1)
        for (int blk_i = 0; blk_i < nblocks; ++blk_i) {
          const int i0 = blk_i * MB, m = std::min(MB, n - i0);
          const int Mp = int(round_up(m, MR));
          T* x = w.x.data();
          T* P = w.P.data();
          std::memset(x, 0, sizeof(T) * size_t(Mp) * F_ld);
          for (int r = 0; r < m; ++r) {
            const int i = i0 + r;
            const int nbond = nact[i];
            const int32_t* bonds = act.data() + in.off[i];
            const T* blk = geo.data() + goff[i];
            const int nbs = int(round_up(nbond, W));
            T* A = w.blockA.data() + size_t(r) * LL.asz;
            T* xr = x + size_t(r) * F_ld;
            density(LL, w, in, i, blk, nbs, nbond, bonds, Ibuf[tl].data(), A);
            if (gate) fgate[size_t(tl) * n + i] = gate_apply(tl, A);
            gram_forward(LL.B[0], w, A, xr);
            for (int t = 0; t < tl; ++t)
              for (const Block& K : lay[t].B) {
                std::memcpy(P, Abuf[t].data() + size_t(i) * lay[t].asz + K.aoff, sizeof(T) * size_t(K.nbase) * K.Cp);
                products_forward(K, P);
                copy_features(K, P, xr);
              }
          }
          if (squash)
            for (int r = 0; r < m; ++r) {
              T* xr = x + size_t(r) * F_ld;
              T* dxr = w.dx.data() + size_t(r) * F_ld;
              for (int c = 0; c < F; ++c) {
                const T v = (xr[c] - fmean[c]) / fstd[c], u = v / T(tau), q = T(1) + u * u, s = T(1) / std::sqrt(q);
                xr[c] = v * s;
                dxr[c] = s / q / fstd[c];
              }
            }
          double eb[512], hs[512];
          const double tr = t0 ? now() : 0.0;
          readout(w, m, Mp, in.mu + i0, eb, core ? core_c.data() + i0 : nullptr, hs);
          if (t0) timers[5] += now() - tr;
          for (int r = 0; r < m; ++r) {
            const int i = i0 + r;
            const double el = core ? hs[r] * scale : 0.0;      // the learned energy, which the core switch scales
            if (core) eb[r] = el * core_c[i] + e0[in.mu[i]];
            eat[i] += eb[r];
            T* gx = w.gx.data() + size_t(r) * mlp[0].nin_p;
            if (squash)
              for (int c = 0; c < F; ++c) gx[c] *= w.dx[size_t(r) * F_ld + c];
            if (core) {                              // the ZBL energy of the atom also scales the learned one
              const T wz = T(1.0 + el * core_dc[i]);
              for (int b = 0; b < nact[i]; ++b) {
                T* dv = dvec.data() + 3 * size_t(act[in.off[i] + b]);
                dv[0] *= wz;
                dv[1] *= wz;
                dv[2] *= wz;
              }
            }
            for (int t = 0; t < tl; ++t)          // the earlier layers' feature gradients, as rows
              for (const Block& K : lay[t].B) {
                T* gf = gfeat[t].data() + size_t(i) * lay[t].gsz + K.goff;
                std::memset(gf, 0, sizeof(T) * K.frows.size() * K.Cp);
                for (size_t f = 0; f < K.frows.size(); ++f)
                  std::memcpy(gf + f * K.Cp, gx + K.foff + f * K.C, sizeof(T) * K.C);
              }
            const T* A = w.blockA.data() + size_t(r) * LL.asz;
            gram_back(LL.B[0], w, A, gx, gAbuf[tl].data() + size_t(i) * LL.asz);
            if (gate) gate_back(tl, fgate[size_t(tl) * n + i], A, gAbuf[tl].data() + size_t(i) * LL.asz, 0.0);
          }
        }
        if (t0) clk[3] = now();
        // BACK: layers L-2 .. 0
        for (int t = L - 2; t >= 0; --t) {
#pragma omp for schedule(dynamic, chunk)
          for (int k = 0; k < n; ++k) back_atom(w, in, t, k, nact[k], act.data() + in.off[k]);
        }
        if (t0) clk[4] = now();
        // FORCE: dE/d(bond) of the atom's bonds minus that of their reverses; virial
        double lv[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma omp for schedule(static)
        for (int k = 0; k < n; ++k) {
          const int nbond = nact[k];
          const int32_t* bonds = act.data() + in.off[k];
          double fx = 0, fy = 0, fz = 0;
          for (int b = 0; b < nbond; ++b) {
            const int c = bonds[b], j = in.cj[c];
            const T* d = dvec.data() + 3 * size_t(c);
            const T* dr = dvec.data() + 3 * size_t(in.rev[c]);
            fx += double(d[0]) - double(dr[0]);
            fy += double(d[1]) - double(dr[1]);
            fz += double(d[2]) - double(dr[2]);
            const double* sh = in.csh + 3 * size_t(c);
            const double v[3] = {in.pos[3 * j] - in.pos[3 * k] + sh[0], in.pos[3 * j + 1] - in.pos[3 * k + 1] + sh[1],
                                 in.pos[3 * j + 2] - in.pos[3 * k + 2] + sh[2]};
            for (int a = 0; a < 3; ++a)
              for (int bb = 0; bb < 3; ++bb) lv[3 * a + bb] += v[a] * double(d[bb]);
          }
          forces[3 * k] = fx;
          forces[3 * k + 1] = fy;
          forces[3 * k + 2] = fz;
        }
#pragma omp critical
        for (int a = 0; a < 9; ++a) vir[a] += lv[a];
      } catch (const std::exception& e) {
#pragma omp critical
        err = e.what();
      }
    }
    if (!err.empty()) throw std::runtime_error(err);
    clk[5] = now();
    for (int k = 0; k < 5; ++k) timers[k] += clk[k + 1] - clk[k];
    double E = 0.0;
    int64_t nbt = 0;
    for (int i = 0; i < n; ++i) {
      E += eat[i];
      nbt += nact[i];
    }
    *energy = E;
    *nbonds = nbt;
    if (e_atoms) std::memcpy(e_atoms, eat.data(), sizeof(double) * n);
    std::memcpy(virial, vir, sizeof vir);
    return 0;
  }
};

template <typename T> struct Handle {
  Engine<T> eng;
};

void set_err(char* err, int len, const char* msg) {
  if (err && len > 0) {
    std::strncpy(err, msg, size_t(len) - 1);
    err[len - 1] = '\0';
  }
}

template <typename T>
void* create(const int64_t* I, int64_t nI, const double* D, int64_t nD, char* err, int len) {
  try {
    std::unique_ptr<Handle<T>> h(new Handle<T>());
    Reader rd{I, nI, 0, D, nD, 0};
    h->eng.load(rd);
    return h.release();
  } catch (const std::exception& e) {
    set_err(err, len, e.what());
    return nullptr;
  }
}

template <typename T>
int compute(void* hp, int n, const double* pos, const int32_t* mu, const int64_t* off, const int32_t* cj,
            const double* csh, const int32_t* rev, double* energy, double* e_atoms, double* forces, double* virial,
            int64_t* nbonds, int nthreads, char* err, int len) {
  try {
    auto* h = static_cast<Handle<T>*>(hp);
    typename Engine<T>::Input in{n, pos, mu, off, cj, csh, rev};
    return h->eng.compute(in, energy, e_atoms, forces, virial, nbonds, nthreads);
  } catch (const std::exception& e) {
    set_err(err, len, e.what());
    return 1;
  }
}

}  // namespace

extern "C" {
void* vv_create_f32(const int64_t* I, int64_t nI, const double* D, int64_t nD, char* err, int len) {
  return create<float>(I, nI, D, nD, err, len);
}
void* vv_create_f64(const int64_t* I, int64_t nI, const double* D, int64_t nD, char* err, int len) {
  return create<double>(I, nI, D, nD, err, len);
}
void vv_free_f32(void* h) { delete static_cast<Handle<float>*>(h); }
void vv_free_f64(void* h) { delete static_cast<Handle<double>*>(h); }
int vv_compute_f32(void* h, int n, const double* pos, const int32_t* mu, const int64_t* off, const int32_t* cj,
                   const double* csh, const int32_t* rev, double* energy, double* e_atoms, double* forces,
                   double* virial, int64_t* nbonds, int nthreads, char* err, int len) {
  return compute<float>(h, n, pos, mu, off, cj, csh, rev, energy, e_atoms, forces, virial, nbonds, nthreads, err, len);
}
int vv_compute_f64(void* h, int n, const double* pos, const int32_t* mu, const int64_t* off, const int32_t* cj,
                   const double* csh, const int32_t* rev, double* energy, double* e_atoms, double* forces,
                   double* virial, int64_t* nbonds, int nthreads, char* err, int len) {
  return compute<double>(h, n, pos, mu, off, cj, csh, rev, energy, e_atoms, forces, virial, nbonds, nthreads, err, len);
}
void vv_timers_f32(void* h, double* out) {
  std::memcpy(out, static_cast<Handle<float>*>(h)->eng.timers, 6 * sizeof(double));
}
void vv_timers_f64(void* h, double* out) {
  std::memcpy(out, static_cast<Handle<double>*>(h)->eng.timers, 6 * sizeof(double));
}
int vv_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}
int vv_version() { return 1; }
}
