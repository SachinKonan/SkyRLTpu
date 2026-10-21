/* Clean-room incremental proxy cost for the Partcl/TILOS macro placement task.
 *
 * Written from the official Python reference only:
 *   external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py
 *     get_cost / get_wirelength, get_grid_cells_density / get_density_cost,
 *     get_routing, __two_pin_net_routing, __three_pin_net_routing, __l_routing,
 *     __t_routing, __split_net, __macro_route_over_grid_cell,
 *     __smooth_routing_cong, get_congestion_cost / abu
 *   macro_place/objective.py (grid-cell location clamped to the grid)
 *
 * proxy = wirelength + 0.5 * density + 0.5 * congestion.
 *
 * Every cost term is a sum of per-net or per-block contributions on fixed arrays,
 * and smoothing is linear, so moving one block updates only that block's nets and
 * cells. The top-k reductions (density top 10%, congestion top 5%) are recomputed
 * on each score with a linear-time selection.
 *
 * Numeric conventions follow the grader: block centers are float32, a macro pin is
 * float32(center + offset), a port keeps its double position, and grid-cell indices
 * of float32 coordinates are computed with float32 division.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n, nh, npin, nnet, rows, cols, ncell, smooth;
    double W, H, gw, gh, grid_area, v_routes, h_routes, halloc, valloc, wl_norm;
    float gwf, ghf;
    double *pos, *size;              /* [n*2] */
    int *pin_owner;                  /* block index, or -1 for a port */
    double *pin_ox, *pin_oy;         /* offset, or absolute position for a port */
    int *net_start;                  /* [nnet+1], first pin of each net is its driver */
    double *weight;                  /* [nnet] */
    int *adj_start, *adj;            /* block -> distinct nets */
    double *hpwl;                    /* [nnet], weighted */
    double total_hpwl;
    double *SV, *SH;                 /* smoothed, normalized net routing */
    double *MV, *MH;                 /* normalized hard-macro routing allocation */
    double *occ;                     /* occupied area per cell */
    /* scratch */
    int *stamp, stamp_id;
    int *gr, *gc;                    /* per-pin cell of the current net */
    int *ur_, *uc_;                  /* distinct cells of the current net */
    double *scratch;                 /* [2*ncell] */
} State;

/* ---------- geometry helpers ---------- */

static inline float f32(double x) { return (float)x; }

static inline void cell_of_f32(const State *s, float x, float y, int *r, int *c) {
    int rr = (int)floorf(y / s->ghf), cc = (int)floorf(x / s->gwf);
    if (rr < 0) rr = 0; if (rr > s->rows - 1) rr = s->rows - 1;
    if (cc < 0) cc = 0; if (cc > s->cols - 1) cc = s->cols - 1;
    *r = rr; *c = cc;
}

static inline void cell_of_f64(const State *s, double x, double y, int *r, int *c) {
    int rr = (int)floor(y / s->gh), cc = (int)floor(x / s->gw);
    if (rr < 0) rr = 0; if (rr > s->rows - 1) rr = s->rows - 1;
    if (cc < 0) cc = 0; if (cc > s->cols - 1) cc = s->cols - 1;
    *r = rr; *c = cc;
}

/* Pin position and grid cell. */
static inline void pin_xy(const State *s, int p, double *x, double *y, int *r, int *c) {
    int o = s->pin_owner[p];
    if (o < 0) {
        *x = s->pin_ox[p]; *y = s->pin_oy[p];
        cell_of_f64(s, *x, *y, r, c);
    } else {
        float fx = f32(s->pos[2*o]) + f32(s->pin_ox[p]);
        float fy = f32(s->pos[2*o+1]) + f32(s->pin_oy[p]);
        *x = fx; *y = fy;
        cell_of_f32(s, fx, fy, r, c);
    }
}

/* Block edges, float32 as in the grader. */
static inline void block_edges(const State *s, int b, double *x0, double *x1, double *y0, double *y1) {
    float x = f32(s->pos[2*b]), y = f32(s->pos[2*b+1]);
    float hw = f32(s->size[2*b] / 2.0), hh = f32(s->size[2*b+1] / 2.0);
    *x0 = (float)(x - hw); *x1 = (float)(x + hw);
    *y0 = (float)(y - hh); *y1 = (float)(y + hh);
}

static inline void block_cells(const State *s, double x0, double x1, double y0, double y1,
                               int *r0, int *r1, int *c0, int *c1) {
    cell_of_f32(s, (float)x1, (float)y1, r1, c1);
    cell_of_f32(s, (float)x0, (float)y0, r0, c0);
}

/* ---------- routing primitives (sign = +1 add, -1 remove) ---------- */

static inline void add_V(State *s, int row, int col, double w) {
    /* V routing is smoothed along columns within the row. */
    int lp = col - s->smooth, rp = col + s->smooth;
    if (lp < 0) lp = 0;
    if (rp > s->cols - 1) rp = s->cols - 1;
    double val = (w / s->v_routes) / (double)(rp - lp + 1);
    double *row_ptr = s->SV + (size_t)row * s->cols;
    for (int k = lp; k <= rp; k++) row_ptr[k] += val;
}

static inline void add_H(State *s, int row, int col, double w) {
    /* H routing is smoothed along rows within the column. */
    int lp = row - s->smooth, up = row + s->smooth;
    if (lp < 0) lp = 0;
    if (up > s->rows - 1) up = s->rows - 1;
    double val = (w / s->h_routes) / (double)(up - lp + 1);
    for (int k = lp; k <= up; k++) s->SH[(size_t)k * s->cols + col] += val;
}

static void two_pin(State *s, int sr, int sc, int kr, int kc, double w) {
    int cmin = sc < kc ? sc : kc, cmax = sc < kc ? kc : sc;
    int rmin = sr < kr ? sr : kr, rmax = sr < kr ? kr : sr;
    for (int col = cmin; col < cmax; col++) add_H(s, sr, col, w);
    for (int row = rmin; row < rmax; row++) add_V(s, row, kc, w);
}

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

/* sort three (row, col) cells by key (col, row) */
static void sort3_colrow(int *y, int *x) {
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2 - i; j++)
            if (x[j] > x[j+1] || (x[j] == x[j+1] && y[j] > y[j+1])) {
                int t = x[j]; x[j] = x[j+1]; x[j+1] = t;
                t = y[j]; y[j] = y[j+1]; y[j+1] = t;
            }
}

/* sort three (row, col) cells by key (row, col) */
static void sort3_rowcol(int *y, int *x) {
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2 - i; j++)
            if (y[j] > y[j+1] || (y[j] == y[j+1] && x[j] > x[j+1])) {
                int t = x[j]; x[j] = x[j+1]; x[j+1] = t;
                t = y[j]; y[j] = y[j+1]; y[j+1] = t;
            }
}

static void three_pin(State *s, const int *rows, const int *cols, double w) {
    int y[3] = {rows[0], rows[1], rows[2]}, x[3] = {cols[0], cols[1], cols[2]};
    sort3_colrow(y, x);
    int y1 = y[0], x1 = x[0], y2 = y[1], x2 = x[1], y3 = y[2], x3 = x[2];
    if (x1 < x2 && x2 < x3 && imin(y1, y3) < y2 && imax(y1, y3) > y2) {
        /* L routing on the same (col, row) order */
        for (int col = x1; col < x2; col++) add_H(s, y1, col, w);
        for (int col = x2; col < x3; col++) add_H(s, y2, col, w);
        for (int row = imin(y1, y2); row < imax(y1, y2); row++) add_V(s, row, x2, w);
        for (int row = imin(y2, y3); row < imax(y2, y3); row++) add_V(s, row, x3, w);
    } else if (x2 == x3 && x1 < x2 && y1 < imin(y2, y3)) {
        for (int col = x1; col < x2; col++) add_H(s, y1, col, w);
        for (int row = y1; row < imax(y2, y3); row++) add_V(s, row, x2, w);
    } else if (y2 == y3) {
        for (int col = x1; col < x2; col++) add_H(s, y1, col, w);
        for (int col = x2; col < x3; col++) add_H(s, y2, col, w);
        for (int row = imin(y2, y1); row < imax(y2, y1); row++) add_V(s, row, x2, w);
    } else {
        /* T routing: cells sorted by (row, col) */
        sort3_rowcol(y, x);
        y1 = y[0]; x1 = x[0]; y2 = y[1]; x2 = x[1]; y3 = y[2]; x3 = x[2];
        int xmin = imin(x1, imin(x2, x3)), xmax = imax(x1, imax(x2, x3));
        for (int col = xmin; col < xmax; col++) add_H(s, y2, col, w);
        for (int row = imin(y1, y2); row < imax(y1, y2); row++) add_V(s, row, x1, w);
        for (int row = imin(y2, y3); row < imax(y2, y3); row++) add_V(s, row, x3, w);
    }
}

/* Add (sign=+1) or remove (sign=-1) one net's HPWL and routing at current positions. */
static void net_update(State *s, int j, double sign) {
    int a = s->net_start[j], z = s->net_start[j+1];
    double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    int nd = 0;
    s->stamp_id++;
    if (s->stamp_id == 0x7fffffff) { memset(s->stamp, 0, sizeof(int) * s->ncell); s->stamp_id = 1; }
    for (int p = a; p < z; p++) {
        double x, y; int r, c;
        pin_xy(s, p, &x, &y, &r, &c);
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        int cell = r * s->cols + c;
        if (s->stamp[cell] != s->stamp_id) {
            s->stamp[cell] = s->stamp_id;
            s->ur_[nd] = r; s->uc_[nd] = c; nd++;
        }
    }
    double h = s->weight[j] * ((xmax - xmin) + (ymax - ymin));
    if (sign > 0) { s->hpwl[j] = h; s->total_hpwl += h; }
    else { s->total_hpwl -= s->hpwl[j]; s->hpwl[j] = 0.0; }

    /* get_routing: a port driver routes with weight 1; a macro pin uses its weight when > 1 */
    double w = ((s->pin_owner[a] >= 0 && s->weight[j] > 1.0) ? s->weight[j] : 1.0) * sign;
    int sr = s->ur_[0], sc = s->uc_[0];   /* driver cell is the first distinct cell */
    if (nd == 2) {
        two_pin(s, sr, sc, s->ur_[1], s->uc_[1], w);
    } else if (nd == 3) {
        three_pin(s, s->ur_, s->uc_, w);
    } else if (nd > 3) {
        for (int k = 1; k < nd; k++) two_pin(s, sr, sc, s->ur_[k], s->uc_[k], w);
    }
}

/* Add or remove one block's density area and, for a hard macro, its routing allocation. */
static void block_update(State *s, int b, double sign) {
    double x0, x1, y0, y1; int r0, r1, c0, c1;
    block_edges(s, b, &x0, &x1, &y0, &y1);
    block_cells(s, x0, x1, y0, y1, &r0, &r1, &c0, &c1);
    int hard = b < s->nh;
    int pv = 0, ph = 0;
    for (int r = r0; r <= r1; r++) {
        double cy0 = r * s->gh, cy1 = (r + 1) * s->gh;
        double ylo = y0 > cy0 ? y0 : cy0, yhi = y1 < cy1 ? y1 : cy1, dy = yhi - ylo;
        for (int c = c0; c <= c1; c++) {
            double cx0 = c * s->gw, cx1 = (c + 1) * s->gw;
            double xlo = x0 > cx0 ? x0 : cx0, xhi = x1 < cx1 ? x1 : cx1, dx = xhi - xlo;
            int cell = r * s->cols + c;
            /* density: __overlap_area counts touching (>= 0) overlaps */
            if (dx >= 0 && dy >= 0) s->occ[cell] += sign * dx * dy;
            if (hard) {
                /* routing allocation: __overlap_dist needs strict overlap */
                double xd = (dx > 0 && dy > 0) ? dx : 0.0, yd = (dx > 0 && dy > 0) ? dy : 0.0;
                if (r1 != r0 && (r == r0 || r == r1) && fabs(yd - s->gh) > 1e-5) pv = 1;
                if (c1 != c0 && (c == c0 || c == c1) && fabs(xd - s->gw) > 1e-5) ph = 1;
                s->MV[cell] += sign * xd * s->valloc / s->v_routes;
                s->MH[cell] += sign * yd * s->halloc / s->h_routes;
            }
        }
    }
    if (!hard) return;
    if (pv) {
        int r = r1; double cy0 = r * s->gh, cy1 = (r + 1) * s->gh;
        double ylo = y0 > cy0 ? y0 : cy0, yhi = y1 < cy1 ? y1 : cy1, dy = yhi - ylo;
        for (int c = c0; c <= c1; c++) {
            double cx0 = c * s->gw, cx1 = (c + 1) * s->gw;
            double xlo = x0 > cx0 ? x0 : cx0, xhi = x1 < cx1 ? x1 : cx1, dx = xhi - xlo;
            double xd = (dx > 0 && dy > 0) ? dx : 0.0;
            s->MV[r * s->cols + c] -= sign * xd * s->valloc / s->v_routes;
        }
    }
    if (ph) {
        int c = c1; double cx0 = c * s->gw, cx1 = (c + 1) * s->gw;
        double xlo = x0 > cx0 ? x0 : cx0, xhi = x1 < cx1 ? x1 : cx1, dx = xhi - xlo;
        for (int r = r0; r <= r1; r++) {
            double cy0 = r * s->gh, cy1 = (r + 1) * s->gh;
            double ylo = y0 > cy0 ? y0 : cy0, yhi = y1 < cy1 ? y1 : cy1, dy = yhi - ylo;
            double yd = (dx > 0 && dy > 0) ? dy : 0.0;
            s->MH[r * s->cols + c] -= sign * yd * s->halloc / s->h_routes;
        }
    }
}

/* ---------- top-k reductions ---------- */

static int cmp_desc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

/* Sum of the k largest values of v[0..m) (k <= m), via quickselect. Scrambles v. */
static double topk_sum(double *v, int m, int k) {
    if (k <= 0) return 0.0;
    if (k >= m) { double t = 0; for (int i = 0; i < m; i++) t += v[i]; return t; }
    int lo = 0, hi = m - 1, target = k - 1;      /* place k-th largest at index target */
    while (lo < hi) {
        double pivot = v[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] > pivot) i++;
            while (v[j] < pivot) j--;
            if (i <= j) { double t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }
    /* v[0..k) now holds the k largest (in any order); sum them sorted for stability */
    qsort(v, (size_t)k, sizeof(double), cmp_desc);
    double t = 0; for (int i = 0; i < k; i++) t += v[i];
    return t;
}

static double density_cost(State *s) {
    int m = 0;
    for (int i = 0; i < s->ncell; i++) {
        double d = s->occ[i] / s->grid_area;
        if (d != 0.0 && fabs(d) > 1e-12) s->scratch[m++] = d;
    }
    if (s->ncell < 10) {
        if (m == 0) return 0.0;
        double t = 0; for (int i = 0; i < m; i++) t += s->scratch[i];
        return 0.5 * t / m;
    }
    int cnt = (int)floor(s->ncell * 0.1);
    if (cnt == 0) return 0.0;
    double t = topk_sum(s->scratch, m, cnt < m ? cnt : m);
    return 0.5 * t / cnt;
}

static double congestion_cost(State *s) {
    int m = 2 * s->ncell;
    for (int i = 0; i < s->ncell; i++) {
        s->scratch[i] = s->SV[i] + s->MV[i];
        s->scratch[s->ncell + i] = s->SH[i] + s->MH[i];
    }
    int cnt = (int)floor(m * 0.05);
    if (cnt == 0) {
        double mx = -INFINITY; for (int i = 0; i < m; i++) if (s->scratch[i] > mx) mx = s->scratch[i];
        return mx;
    }
    return topk_sum(s->scratch, m, cnt) / cnt;
}

/* ---------- public API ---------- */

void mp_destroy(State *s);

static void rebuild(State *s) {
    memset(s->SV, 0, sizeof(double) * s->ncell); memset(s->SH, 0, sizeof(double) * s->ncell);
    memset(s->MV, 0, sizeof(double) * s->ncell); memset(s->MH, 0, sizeof(double) * s->ncell);
    memset(s->occ, 0, sizeof(double) * s->ncell);
    s->total_hpwl = 0.0;
    for (int j = 0; j < s->nnet; j++) net_update(s, j, +1.0);
    for (int b = 0; b < s->n; b++) block_update(s, b, +1.0);
}

State *mp_create(int n, int nh, const double *pos, const double *size,
                 int npin, const int *pin_owner, const double *pin_ox, const double *pin_oy,
                 int nnet, const int *net_start, const double *weight,
                 int rows, int cols, double W, double H, double h_rpm, double v_rpm,
                 double halloc, double valloc, int smooth, double wl_norm) {
    State *s = calloc(1, sizeof(State));
    if (!s) return NULL;
    s->n = n; s->nh = nh; s->npin = npin; s->nnet = nnet; s->rows = rows; s->cols = cols;
    s->ncell = rows * cols; s->smooth = smooth; s->W = W; s->H = H;
    s->gw = W / cols; s->gh = H / rows; s->gwf = (float)s->gw; s->ghf = (float)s->gh;
    s->grid_area = s->gw * s->gh;
    s->v_routes = s->gw * v_rpm; s->h_routes = s->gh * h_rpm;
    s->halloc = halloc; s->valloc = valloc; s->wl_norm = wl_norm;
#define ALLOC(p, cnt, T) do { p = (T *)malloc(sizeof(T) * (size_t)((cnt) > 0 ? (cnt) : 1)); if (!p) { mp_destroy(s); return NULL; } } while (0)
    ALLOC(s->pos, 2*n, double); ALLOC(s->size, 2*n, double);
    ALLOC(s->pin_owner, npin, int); ALLOC(s->pin_ox, npin, double); ALLOC(s->pin_oy, npin, double);
    ALLOC(s->net_start, nnet + 1, int); ALLOC(s->weight, nnet, double); ALLOC(s->hpwl, nnet, double);
    ALLOC(s->SV, s->ncell, double); ALLOC(s->SH, s->ncell, double);
    ALLOC(s->MV, s->ncell, double); ALLOC(s->MH, s->ncell, double); ALLOC(s->occ, s->ncell, double);
    ALLOC(s->scratch, 2 * s->ncell, double);
    ALLOC(s->ur_, npin, int); ALLOC(s->uc_, npin, int); ALLOC(s->gr, 1, int); ALLOC(s->gc, 1, int);
    s->stamp = calloc((size_t)s->ncell, sizeof(int)); if (!s->stamp) { mp_destroy(s); return NULL; }
    memcpy(s->pos, pos, sizeof(double) * 2 * n); memcpy(s->size, size, sizeof(double) * 2 * n);
    memcpy(s->pin_owner, pin_owner, sizeof(int) * npin);
    memcpy(s->pin_ox, pin_ox, sizeof(double) * npin); memcpy(s->pin_oy, pin_oy, sizeof(double) * npin);
    memcpy(s->net_start, net_start, sizeof(int) * (nnet + 1)); memcpy(s->weight, weight, sizeof(double) * nnet);
    /* block -> distinct nets */
    int *count = calloc((size_t)n + 1, sizeof(int)), *last = malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    if (!count || !last) { free(count); free(last); mp_destroy(s); return NULL; }
    for (int b = 0; b < n; b++) last[b] = -1;
    for (int j = 0; j < nnet; j++)
        for (int p = net_start[j]; p < net_start[j+1]; p++) {
            int o = pin_owner[p];
            if (o >= 0 && last[o] != j) { last[o] = j; count[o]++; }
        }
    ALLOC(s->adj_start, n + 1, int);
    s->adj_start[0] = 0;
    for (int b = 0; b < n; b++) s->adj_start[b+1] = s->adj_start[b] + count[b];
    ALLOC(s->adj, s->adj_start[n], int);
    for (int b = 0; b < n; b++) { last[b] = -1; count[b] = 0; }
    for (int j = 0; j < nnet; j++)
        for (int p = net_start[j]; p < net_start[j+1]; p++) {
            int o = pin_owner[p];
            if (o >= 0 && last[o] != j) { last[o] = j; s->adj[s->adj_start[o] + count[o]++] = j; }
        }
    free(count); free(last);
    rebuild(s);
    return s;
}

void mp_destroy(State *s) {
    if (!s) return;
    free(s->pos); free(s->size); free(s->pin_owner); free(s->pin_ox); free(s->pin_oy);
    free(s->net_start); free(s->weight); free(s->adj_start); free(s->adj); free(s->hpwl);
    free(s->SV); free(s->SH); free(s->MV); free(s->MH); free(s->occ); free(s->scratch);
    free(s->stamp); free(s->gr); free(s->gc); free(s->ur_); free(s->uc_);
    free(s);
}

void mp_rebuild(State *s) { rebuild(s); }

void mp_set_positions(State *s, const double *pos) {
    memcpy(s->pos, pos, sizeof(double) * 2 * s->n);
    rebuild(s);
}

/* Move one block; incremental update of every term. */
void mp_move(State *s, int b, double x, double y) {
    for (int k = s->adj_start[b]; k < s->adj_start[b+1]; k++) net_update(s, s->adj[k], -1.0);
    block_update(s, b, -1.0);
    s->pos[2*b] = x; s->pos[2*b+1] = y;
    for (int k = s->adj_start[b]; k < s->adj_start[b+1]; k++) net_update(s, s->adj[k], +1.0);
    block_update(s, b, +1.0);
}

/* out = {proxy, wirelength, density, congestion} */
void mp_score(State *s, double *out) {
    double wl = s->total_hpwl / s->wl_norm;
    double den = density_cost(s);
    double cong = congestion_cost(s);
    out[0] = wl + 0.5 * den + 0.5 * cong;
    out[1] = wl; out[2] = den; out[3] = cong;
}
