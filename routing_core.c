/* Kaskade nichtlinearer Speicher (Kalinin-Miljukov-Art) mit abflussabhängiger Wellengeschwindigkeit c(Q).
 *
 * Eine Strecke der Länge L wird in n gleiche Teilstrecken geteilt. Jede ist ein Speicher S = K * O mit K = dx / c(Qref),
 * Qref = Mittel aus Zufluss (Zeitschritt-Mitte) und Abfluss zu Schrittbeginn. Der Zufluss ändert sich im Schritt linear;
 * dafür ist die Lösung des linearen Speichers exakt:
 *   O1 = O0 * a + I0 * (1 - a) + (I1 - I0) * (1 - K * (1 - a) / dt),  a = exp(-dt / K)
 * Verzögerung der Strecke bei Q: n * K = L / c(Q). Mehr Teilstrecken (n) = weniger Abflachung der Welle.
 * c(Q): lineare Interpolation von log c über log Q an nn Stützstellen (außerhalb konstant).
 */
#include <math.h>

static double celerity(double q, int nn, const double *lq, const double *lc)
{
    double x = log(q > 1e-3 ? q : 1e-3);
    double v;
    if (nn == 1 || x <= lq[0]) v = lc[0];
    else if (x >= lq[nn - 1]) v = lc[nn - 1];
    else {
        int k = 0;
        while (k < nn - 2 && x > lq[k + 1]) k++;
        double f = (x - lq[k]) / (lq[k + 1] - lq[k]);
        v = lc[k] + f * (lc[k + 1] - lc[k]);
    }
    v = exp(v);
    if (v < 0.05) v = 0.05;
    if (v > 12.0) v = 12.0;
    return v;
}

/* in, out: E Reihen der Länge T (zeilenweise). Anfangszustand: stationär beim ersten Zuflusswert. */
void cascade(const double *in, double *out, int E, int T, int n, double length_km, double dt_h,
             int nn, const double *lq, const double *lc)
{
    double dx = length_km / n;
    double O[256];
    if (n > 256) n = 256;
    for (int e = 0; e < E; e++) {
        const double *x = in + (long)e * T;
        double *y = out + (long)e * T;
        for (int j = 0; j < n; j++) O[j] = x[0];
        y[0] = x[0];
        for (int t = 1; t < T; t++) {
            double i0 = x[t - 1], i1 = x[t];
            for (int j = 0; j < n; j++) {
                double qref = 0.5 * (0.5 * (i0 + i1) + O[j]);
                double c = celerity(qref, nn, lq, lc);
                double K = dx / (3.6 * c);
                double em = -expm1(-dt_h / K);          /* 1 - a */
                double a = 1.0 - em;
                double o0 = O[j];
                double o1 = o0 * a + i0 * em + (i1 - i0) * (1.0 - K * em / dt_h);
                i0 = o0;
                i1 = o1;
                O[j] = o1;
            }
            y[t] = O[n - 1];
        }
    }
}

/* Zwei nichtlineare Speicher in Reihe (lateral._nl_stores, gleiche Rechnung): Speicherzeit K(S) = K / (1 + S / (K oref))^a, Zufluss im Schritt
 * konstant, Abfluss aus der Bilanz. peff, out: E Reihen der Länge T. */
void nl_stores(const double *peff, double *out, int E, int T, double k1, double k2, double dt_h, double oref, double a)
{
    for (int e = 0; e < E; e++) {
        const double *p = peff + (long)e * T;
        double *y = out + (long)e * T;
        double s1 = k1 * p[0], s2 = k2 * p[0];
        y[0] = p[0];
        for (int t = 1; t < T; t++) {
            double kt1 = k1 / pow(1.0 + s1 / (k1 * oref), a);
            double e1 = exp(-dt_h / kt1);
            double s1n = s1 * e1 + p[t] * kt1 * (1.0 - e1);
            double i2 = p[t] - (s1n - s1) / dt_h;
            s1 = s1n;
            double kt2 = k2 / pow(1.0 + s2 / (k2 * oref), a);
            double e2 = exp(-dt_h / kt2);
            double s2n = s2 * e2 + i2 * kt2 * (1.0 - e2);
            y[t] = i2 - (s2n - s2) / dt_h;
            s2 = s2n;
        }
    }
}

/* Zwei lineare Speicher in Reihe (lateral._two_stores ohne nl, gleiche Rechnung). */
void two_stores(const double *peff, double *out, int E, int T, double k1, double k2, double dt_h)
{
    double a1 = exp(-dt_h / k1), a2 = exp(-dt_h / k2), w2 = 1.0 - k2 * (1.0 - a2) / dt_h;
    for (int e = 0; e < E; e++) {
        const double *p = peff + (long)e * T;
        double *y = out + (long)e * T;
        double o1 = p[0], o2 = p[0];
        y[0] = p[0];
        for (int t = 1; t < T; t++) {
            double o1n = a1 * o1 + (1.0 - a1) * p[t];
            o2 = a2 * o2 + (1.0 - a2) * o1 + (o1n - o1) * w2;
            o1 = o1n;
            y[t] = o2;
        }
    }
}

/* Vorregenindex (rain.api_series): acc = acc * k + p * dt, fehlende Werte (NaN) zählen als 0. */
void api_series(const double *p, double *out, int T, double k, double dt_h)
{
    double acc = 0.0;
    for (int t = 0; t < T; t++) {
        double v = p[t];
        if (v != v) v = 0.0;
        acc = acc * k + v * dt_h;
        out[t] = acc;
    }
}
