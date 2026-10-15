# Resonance sampling

A numerical study of how observation placement affects the recovery of sharply localized structure. A lightly damped mechanical system provides a concrete example: a response can be smooth almost everywhere while a narrow resonance contains its largest amplitude. The repository connects this sampling problem to random-feature placement and Bayesian readout uncertainty through three executable calculations.

The central experiment compares a 25-point frequency sweep with a dense reference evaluation. Two additional checks examine whether scaling random features actually moves their centres, and whether uncertainty survives when feature parameters are fixed.

## A controlled frequency-response problem

The simulator is a dimensionless, 48-degree-of-freedom fixed-free chain with unit masses and nearest-neighbour springs. Its mass matrix is $M=I$. The stiffness matrix has diagonal entries 2 except for the final entry 1, and off-diagonal entries $-1$. Rayleigh damping is

$$
C=\alpha M+\beta K,\qquad \alpha=0.001,\quad\beta=0.0002.
$$

A harmonic force is applied to the final mass, and displacement is observed at that same location. Writing $b=e_{48}$, the complex response is

$$
H(\omega)=b^\mathsf{T}(K-\omega^2M+i\omega C)^{-1}b.
$$

The reported amplitude is $|H(\omega)|$. The example is useful because its response can be evaluated in two independent algebraic forms. With $K=V\Lambda V^\mathsf{T}$,

$$
H(\omega)=\sum_{j=1}^{48}
\frac{(v_j^\mathsf{T}b)^2}
{\lambda_j-\omega^2+i\omega(\alpha+\beta\lambda_j)}.
$$

A direct complex linear solve and the modal formula are compared at four frequencies, with relative and absolute tolerances of $10^{-9}$. This check separates an error in the response evaluator from an effect of the sampling grid. The modal decomposition is an evaluator-side reference; an eventual query policy should not receive its hidden peak locations.

## What the coarse sweep observes

Both grids cover $[0.01,1]$. The uniform coarse grid contains 25 frequencies, while the reference contains 30,001.

| Quantity | Recorded value |
|---|---:|
| Maximum amplitude on the coarse grid | 52.1951 |
| Maximum amplitude on the reference grid | 1273.0793 |
| Frequency of the reference-grid maximum | 0.032374 |
| Coarse maximum / reference maximum | 0.0409991 |

The coarse sweep reaches only **4.10%** of the reference-grid maximum. The comparison exposes the difference between having observations throughout a domain and actually resolving its narrow features. The denominator is a dense-grid maximum, not a certified continuous optimum.

All values are stored in [checks.json](checks.json). The [response CSV](toy_frequency_response.csv) exports every tenth point of the dense calculation, giving 3,001 display points; peak statistics in the JSON use all 30,001 points.

## Random-feature geometry

For a feature $\tanh(ax+b)$, the centre is $c=-b/a$ when $a\ne0$. Multiplying both $a$ and $b$ by the same nonzero constant leaves this centre unchanged, even though it changes transition width. The script verifies that identity on 100,000 Gaussian draws using a common factor of three.

For independent standard-normal $a$ and $b$, the ratio $-b/a$ has a standard Cauchy distribution. Hence, for $\epsilon>0$,

$$
\Pr(|c|\leq\epsilon)=\frac{2}{\pi}\arctan(\epsilon).
$$

At $\epsilon=0.05$, a single centre falls in the interval with probability approximately 0.03180. For twenty independent features, the probability that at least one centre does so is

$$
1-(1-0.0318045)^{20}\approx0.47608.
$$

This calculation distinguishes **where a feature acts** from **how steeply it acts**. A common parameter-scale adjustment cannot by itself repair a centre-placement problem. A centre inside the interval is also only a geometric condition: approximation quality depends on widths, readout weights, and the target function.

## Conditional uncertainty at fixed features

A scalar Gaussian readout gives an exact check. Use design values $(1,2)$, targets $(1,2)$, unit noise variance, and a unit-variance Gaussian prior on the readout coefficient $w$. Then

$$
\mathrm{Var}(w\mid D)=\frac{1}{1+1^2+2^2}=\frac16,\qquad
\mathbb E[w\mid D]=\frac{1+4}{6}=\frac56.
$$

At test feature value 3, the posterior function mean is 2.5 and its variance is $3^2/6=1.5$. Adding observation noise gives predictive variance 2.5. Repeating only the conditional mean at those fixed features would produce variance zero.

This small example illustrates why a distribution over fitted feature parameters and a conditional distribution over readout weights are different sources of uncertainty. The distinction becomes consequential when uncertainty is used to select the next observation.

## Run and inspect

~~~bash
python3 -m pip install -r requirements.txt
python3 frequency_response.py
~~~

[frequency_response.py](frequency_response.py) contains the complete simulator, cross-checks, and output generation. It uses NumPy, a fixed random seed, and dimensionless quantities. Running it recreates the JSON and CSV beside the script.

## A next experiment

The simulator supports a prospective comparison of uniform sampling, adaptive interpolation, and uncertainty-guided queries under the same observation budget. Peak-location error, peak-amplitude error, and integrated response error would test different aspects of recovery. Restricting each method to its queried responses would keep the full modal solution available only for evaluation.

The calculations in this repository establish the response and algebraic checks for that comparison. They do not include a trained acquisition policy or measured gains from adaptive sampling.
