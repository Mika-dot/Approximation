using System.Globalization;
using ApproximationLib;

if (args.Contains("--help", StringComparer.OrdinalIgnoreCase))
{
    Console.WriteLine("""
Approximation — symbolic regression from a CSV file

Demo:
  dotnet run --project Approximation -- --report report.html

CSV:
  dotnet run --project Approximation -- --csv data.csv --target y --report report.html

Options:
  --csv PATH          CSV file with a header (comma, semicolon or tab)
  --target NAME       target column (required with --csv)
  --delimiter CHAR    override delimiter detection
  --report [PATH]     generate a standalone HTML report
  --seed NUMBER       reproducible random seed (default: 2026)
  --auto              tune search budget from the dataset shape
""");
    return;
}

Dictionary<string, string?> options = ParseOptions(args);
int seed = options.TryGetValue("--seed", out string? seedText) && int.TryParse(seedText, out int parsedSeed) ? parsedSeed : 2026;
var config = new Config { Seed = seed, EnableLogging = true };

if (options.ContainsKey("--report"))
{
    config.GenerateReport = true;
    if (!string.IsNullOrWhiteSpace(options["--report"])) config.OutputPath = options["--report"]!;
}

double[][] features;
double[] targets;
string[] featureNames;

if (options.TryGetValue("--csv", out string? csvPath) && !string.IsNullOrWhiteSpace(csvPath))
{
    if (!options.TryGetValue("--target", out string? targetName) || string.IsNullOrWhiteSpace(targetName))
        throw new ArgumentException("--target is required when --csv is used.");
    char? delimiter = options.TryGetValue("--delimiter", out string? delimiterText) && !string.IsNullOrEmpty(delimiterText)
        ? delimiterText[0] : null;
    (features, targets, featureNames) = ReadCsv(csvPath, targetName, delimiter);
}
else
{
    double[] x = Enumerable.Range(-30, 61).Select(i => i / 10.0).ToArray();
    features = x.Select(value => new[] { value }).ToArray();
    targets = x.Select(value => value * value - 2 * value + 1).ToArray();
    featureNames = new[] { "x" };
}

Console.WriteLine($"Rows: {features.Length}; features: {string.Join(", ", featureNames)}");
var regressor = new SymbolicRegressor(config);
Result result = options.ContainsKey("--auto")
    ? regressor.AutoFit(features, targets, featureNames)
    : regressor.Fit(features, targets, featureNames);

Console.WriteLine();
Console.WriteLine($"Formula : {result.BestFormula}");
Console.WriteLine($"R²      : {result.R2:F8}");
Console.WriteLine($"RMSE    : {result.RMSE:G6}");
Console.WriteLine($"MAE     : {result.MAE:G6}");
Console.WriteLine($"Nodes   : {result.Size}");
Console.WriteLine($"Time    : {result.TrainingTime.TotalSeconds:F2} s");
if (config.GenerateReport) Console.WriteLine($"Report  : {Path.GetFullPath(config.OutputPath)}");

static Dictionary<string, string?> ParseOptions(string[] args)
{
    var result = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < args.Length; i++)
    {
        if (!args[i].StartsWith("--", StringComparison.Ordinal)) continue;
        string key = args[i];
        string? value = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal) ? args[++i] : null;
        result[key] = value;
    }
    return result;
}

static (double[][] Features, double[] Targets, string[] FeatureNames) ReadCsv(string path, string targetName, char? explicitDelimiter)
{
    string[] lines = File.ReadAllLines(path).Where(line => !string.IsNullOrWhiteSpace(line)).ToArray();
    if (lines.Length < 4) throw new InvalidDataException("CSV must contain a header and at least three data rows.");
    char delimiter = explicitDelimiter ?? new[] { ',', ';', '\t' }.OrderByDescending(c => lines[0].Count(ch => ch == c)).First();
    string[] header = lines[0].Split(delimiter, StringSplitOptions.TrimEntries);
    int target = Array.FindIndex(header, name => string.Equals(name, targetName, StringComparison.OrdinalIgnoreCase));
    if (target < 0) throw new InvalidDataException($"Target column '{targetName}' was not found.");
    int[] inputColumns = Enumerable.Range(0, header.Length).Where(i => i != target).ToArray();
    var rows = new List<double[]>();
    var y = new List<double>();
    for (int line = 1; line < lines.Length; line++)
    {
        string[] cells = lines[line].Split(delimiter, StringSplitOptions.TrimEntries);
        if (cells.Length != header.Length) throw new InvalidDataException($"Row {line + 1} has {cells.Length} columns; expected {header.Length}.");
        rows.Add(inputColumns.Select(i => double.Parse(cells[i], NumberStyles.Float, CultureInfo.InvariantCulture)).ToArray());
        y.Add(double.Parse(cells[target], NumberStyles.Float, CultureInfo.InvariantCulture));
    }
    return (rows.ToArray(), y.ToArray(), inputColumns.Select(i => header[i]).ToArray());
}
