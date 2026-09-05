using System.Globalization;
using ApproximationLib;

var tests = new (string Name, Action Run)[]
{
    ("constants are culture independent", CultureIndependentConstants),
    ("protected operators stay finite", ProtectedOperatorsStayFinite),
    ("affine scaling recovers a linear law", RecoversLinearLaw),
    ("multivariate API recovers an interaction", RecoversInteraction),
    ("seed makes runs reproducible", Reproducible),
    ("invalid input is rejected", RejectsInvalidInput),
    ("standalone report is produced", ProducesReport)
};

int failed = 0;
foreach ((string name, Action run) in tests)
{
    try { run(); Console.WriteLine($"PASS  {name}"); }
    catch (Exception error) { failed++; Console.Error.WriteLine($"FAIL  {name}: {error.Message}"); }
}

if (failed > 0) Environment.Exit(1);
Console.WriteLine($"All {tests.Length} tests passed.");

static Config FastConfig(int seed = 7) => new()
{
    PopulationSize = 80,
    MaxGenerations = 12,
    EarlyStoppingPatience = 5,
    Seed = seed,
    ParallelEvaluation = true,
    ValidationFraction = 0.2,
    MaxDepth = 5,
    MaxNodes = 31
};

static void CultureIndependentConstants()
{
    CultureInfo previous = CultureInfo.CurrentCulture;
    try
    {
        CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
        var tree = new ExpressionTree(1.25);
        Near(1.25, tree.Evaluate(0), 1e-14);
    }
    finally { CultureInfo.CurrentCulture = previous; }
}

static void ProtectedOperatorsStayFinite()
{
    var division = new ExpressionTree("protectedDiv", new ExpressionTree(8), new ExpressionTree(0));
    var logarithm = new ExpressionTree("log", new ExpressionTree(-5));
    var exponential = new ExpressionTree("exp", new ExpressionTree(1000));
    Assert(double.IsFinite(division.Evaluate(0)));
    Assert(double.IsFinite(logarithm.Evaluate(0)));
    Assert(double.IsFinite(exponential.Evaluate(0)));
}

static void RecoversLinearLaw()
{
    double[] x = Enumerable.Range(-10, 21).Select(i => (double)i).ToArray();
    double[] y = x.Select(v => 4 * v + 3).ToArray();
    Result result = new SymbolicRegressor(FastConfig()).Fit(x, y);
    Assert(result.R2 > 0.999999999);
    Near(11, result.Function(2), 1e-8);
}

static void RecoversInteraction()
{
    double[][] x = (from a in Enumerable.Range(-2, 5)
                    from b in Enumerable.Range(-2, 5)
                    select new[] { (double)a, (double)b }).ToArray();
    double[] y = x.Select(row => 2 * row[0] * row[1] + 1).ToArray();
    Result result = new SymbolicRegressor(FastConfig()).Fit(x, y, new[] { "a", "b" });
    Assert(result.R2 > 0.999999999);
    Near(13, result.Predict(2, 3), 1e-8);
}

static void Reproducible()
{
    double[] x = Enumerable.Range(-6, 13).Select(i => (double)i).ToArray();
    double[] y = x.Select(v => v * v + 2).ToArray();
    Result first = new SymbolicRegressor(FastConfig(123)).Fit(x, y);
    Result second = new SymbolicRegressor(FastConfig(123)).Fit(x, y);
    Assert(first.BestFormula == second.BestFormula);
    Near(first.MSE, second.MSE, 1e-14);
}

static void RejectsInvalidInput()
{
    bool rejected = false;
    try { _ = new SymbolicRegressor(FastConfig()).Fit(new[] { 1.0, double.NaN, 3.0 }, new[] { 1.0, 2.0, 3.0 }); }
    catch (ArgumentException) { rejected = true; }
    Assert(rejected);
}

static void ProducesReport()
{
    string path = Path.Combine(Path.GetTempPath(), $"approximation-{Guid.NewGuid():N}.html");
    Config config = FastConfig();
    config.GenerateReport = true;
    config.OutputPath = path;
    try
    {
        _ = new SymbolicRegressor(config).Fit(new[] { -2d, -1, 0, 1, 2, 3, 4, 5 }, new[] { -3d, -1, 1, 3, 5, 7, 9, 11 });
        Assert(File.Exists(path));
        string html = File.ReadAllText(path);
        Assert(html.Contains("<svg", StringComparison.Ordinal));
        Assert(!html.Contains("cdnjs.cloudflare.com", StringComparison.OrdinalIgnoreCase));
        Assert(!html.Contains("cdn.jsdelivr.net", StringComparison.OrdinalIgnoreCase));
        Assert(!html.Contains("unpkg.com", StringComparison.OrdinalIgnoreCase));
    }
    finally { if (File.Exists(path)) File.Delete(path); }
}

static void Assert(bool condition)
{
    if (!condition) throw new InvalidOperationException("Assertion failed.");
}

static void Near(double expected, double actual, double tolerance)
{
    if (Math.Abs(expected - actual) > tolerance)
        throw new InvalidOperationException($"Expected {expected}, got {actual}.");
}
