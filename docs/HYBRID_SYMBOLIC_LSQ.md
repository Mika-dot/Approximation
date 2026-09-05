# Hybrid Symbolic Regression + LSQ

Branch: `feature/hybrid-symbolic-lsq`

This branch explores a direct composition of the two ideas in:

- `Mika-dot/Approximation`: target-driven genetic symbolic regression that discovers explicit analytical expressions;
- `Mika-dot/LSQ-Adaptive`: nonlinear representation followed by a rank-aware Moore-Penrose least-squares target solve.

## Why this composition

A naive integration would call the Python package from the .NET library or run both predictors independently and average them. That adds deployment complexity without creating a new learning mechanism.

The useful composition is to let symbolic regression discover nonlinear coordinates and let LSQ solve how those coordinates should be combined.

For independently discovered symbolic functions

```text
phi_1(X), phi_2(X), ..., phi_k(X)
```

the hybrid model is

```text
y_hat(X) = beta_0 + sum_j beta_j * phi_j(X)
```

The `beta` values are obtained in one Moore-Penrose least-squares solve over a held-out blend set.

## Leakage control

The data are deterministically split into two parts:

1. **structure set** — only this part is given to the symbolic searches;
2. **blend set** — symbolic structures are frozen before this part is used to fit the final coefficients.

This matters because fitting the final LSQ coefficients on the same targets used to evolve the basis would make the reported blend quality optimistic.

## Numerical solve

Let `Phi` contain the predictions of the symbolic basis functions on the blend set. Every column is centered and standardized, then the implementation forms

```text
G = Phi^T Phi
c = Phi^T (y - mean(y))
```

and uses the numerical positive eigensystem of `G` to apply its Moore-Penrose pseudoinverse:

```text
beta = G^+ c
```

Eigenvalues below the machine-precision rank threshold are discarded. This is important because different symbolic searches often discover algebraically different but numerically collinear formulas.

The implementation deliberately has no MathNet, NumPy, SciPy or Python runtime dependency; the small symmetric eigensystem is solved inside `ApproximationLib` by a Jacobi rotation method.

## What is preserved

- The final prediction remains an explicit analytical formula.
- The symbolic basis remains available in `HybridResult.BasisFormulas`.
- Coefficients and numerical rank are exposed for diagnostics.
- The algorithm handles duplicate/collinear symbolic bases through the pseudoinverse rather than an unstable ordinary inverse.
- The .NET DLL remains self-contained at the managed-library level.

## What is *not* claimed

This hybrid must not be described as preserving LSQ-Adaptive's strict `one supervised LSQ solve / zero iterative target optimization` training contract.

Symbolic structure discovery in Approximation is itself target-dependent and evolutionary. The LSQ stage is a single direct solve, but the entire hybrid training procedure is not a one-pass LSQ learner.

The correct claim is narrower:

> symbolic regression supplies an interpretable nonlinear basis; a Moore-Penrose LSQ stage optimally recombines that frozen basis on held-out data.

## Current prototype API

```csharp
var symbolic = new Config
{
    PopulationSize = 600,
    MaxGenerations = 250,
    Seed = 42
};

var hybrid = new HybridSymbolicLsqRegressor(new HybridConfig
{
    BasisModels = 6,
    BlendFraction = 0.25,
    Seed = 2026,
    Symbolic = symbolic
});

HybridResult result = hybrid.Fit(features, targets, featureNames);

Console.WriteLine(result.Formula);
Console.WriteLine($"R2 = {result.Metrics.R2:F6}");
Console.WriteLine($"rank = {result.NumericalRank}/{result.BasisFormulas.Count}");
```

## Next experiments

The current prototype deliberately starts with whole-model bases because this uses the existing public `Result.MultiFunction` API and keeps the change isolated. The next high-value experiments are:

1. **Pareto/archive basis extraction** — expose executable candidates from the symbolic Pareto archive, not just complete independent runs. One GP run could then generate tens of candidate nonlinear coordinates.
2. **Rank/novelty pruning before LSQ** — reject basis columns that are almost perfectly correlated on the blend set before solving. This reduces formula size without sacrificing span.
3. **Residual symbolic rounds** — after the first LSQ solve, run a symbolic search against the held-out residual, append only a genuinely new basis, then solve the full coefficient vector again. This is analogous to basis pursuit, while coefficients are always recomputed globally rather than greedily frozen.
4. **Cross-fitted basis generation** — rotate structure/blend folds to use more of the dataset without letting any LSQ target value influence the symbolic formula that produced its feature value.
5. **Complexity-aware LSQ selection** — optimize validation error subject to a total symbolic-node budget, so a tiny gain from a very large basis does not bloat the final equation.
6. **Benchmark matrix** — compare plain symbolic regression, plain LSQ-Adaptive, and the hybrid on the same regression datasets with fixed splits, reporting R2/RMSE, wall time, numerical rank and final expression complexity.

The most promising long-term form is therefore not an ensemble of two finished predictors. It is a two-level learner:

```text
symbolic search -> diverse analytical dictionary -> rank pruning -> global Moore-Penrose solve
```

That creates a model class neither repository currently implements by itself.
