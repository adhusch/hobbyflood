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
