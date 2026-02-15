# NHPP Models

## Mean value function

All NHPP models in this package use the same mean value function:

$$
\Lambda(t) = \omega \, F(t)
$$

where $\omega$ is the total fault count and $F(t)$ is the CDF for the model’s lifetime distribution.

## Model-specific $F(t)$

Below, $\Phi(\cdot)$ is the standard normal CDF. Unless noted, $F(t)=0$ for $t<0$.

### ExponentialNHPP (rate $\beta$)

$$
F(t) = 1 - e^{-\beta t}
$$

### TruncatedNormalNHPP (mean $\mu$, sd $\sigma$)

$$
F(t)=\frac{\Phi\left(\frac{t-\mu}{\sigma}\right)-\Phi\left(\frac{-\mu}{\sigma}\right)}{1-\Phi\left(\frac{-\mu}{\sigma}\right)}\quad (t\ge 0)
$$

### Pareto2NHPP (shape $k$, scale $\lambda$)

$$
F(t)=1-\left(1+\frac{t}{\lambda}\right)^{-k}
$$

### GammaNHPP (shape $k$, rate $\beta$)

$$
F(t)=\frac{\gamma\left(k,\,\beta t\right)}{\Gamma(k)}
$$

### LogNormalNHPP (meanlog $\mu$, sdlog $\sigma$)

$$
F(t)=\Phi\left(\frac{\ln t-\mu}{\sigma}\right)\quad (t>0)
$$

### TruncatedLogisticNHPP (location $m$, scale $s$)

Let $L(t)=\frac{1}{1+e^{-(t-m)/s}}$. Then

$$
F(t)=\frac{L(t)-L(0)}{1-L(0)}\quad (t\ge 0)
$$

### LogLogisticNHPP (locationlog $m$, scalelog $s$)

Let $L(x)=\frac{1}{1+e^{-(x-m)/s}}$. Then

$$
F(t)=L(\ln t)\quad (t>0)
$$

### TruncatedExtremeValueMaxNHPP (Gumbel max; loc $\mu$, scale $\beta$)

Let $G_{\max}(t)=\exp\{-\exp[-(t-\mu)/\beta]\}$. Then

$$
F(t)=\frac{G_{\max}(t)-G_{\max}(0)}{1-G_{\max}(0)}\quad (t\ge 0)
$$

### TruncatedExtremeValueMinNHPP (Gumbel min; loc $\mu$, scale $\beta$)

Let $G_{\min}(t)=1-\exp\{-\exp[(t-\mu)/\beta]\}$. Then

$$
F(t)=\frac{G_{\min}(t)-G_{\min}(0)}{1-G_{\min}(0)}\quad (t\ge 0)
$$

### LogExtremeValueMaxNHPP (log-Gumbel max; loclog $\mu$, scalelog $\beta$)

Let $G_{\max}$ be the max-Gumbel CDF. Then

$$
F(t)=G_{\max}(\ln t)\quad (t>0)
$$

### LogExtremeValueMinNHPP (log-Gumbel min; loclog $\mu$, scalelog $\beta$)

Let $G_{\min}$ be the min-Gumbel CDF. Then

$$
F(t)=G_{\min}(\ln t)\quad (t>0)
$$

### CanonicalPhaseTypeNHPP (CF1; $\alpha$, $\text{rate}$)

CF1 is a phase-type (PH) distribution with generator matrix $T$ having

- diagonal entries $T_{ii}=-\text{rate}_i$
- superdiagonal entries $T_{i,i+1}=\text{rate}_i$

Let $\alpha$ be the initial probability vector and $\mathbf{1}$ the all-ones vector. Then

$$
F(t)=1-\alpha\,e^{T t}\,\mathbf{1}
$$

In code, this corresponds to `pcf1(t, alpha, rate)`.
