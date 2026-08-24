using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;

namespace ApproximationLib;

/// <summary>
/// Genetic-programming symbolic regressor with epsilon-lexicase selection,
/// validation-based early stopping, Pareto model selection and constant tuning.
/// </summary>
public sealed class SymbolicRegressor
{
    private readonly Config _config;
    private readonly Random _rng;
    private readonly List<GenerationData> _history = new();
    private readonly List<ExpressionTree> _archive = new();
    private int _featureCount;
    private string[] _featureNames = Array.Empty<string>();

    public SymbolicRegressor(Config? config = null)
    {
        _config = (config ?? new Config()).Copy();
        _config.Validate();
        _rng = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
    }

    public Result Fit(double[] xData, double[] yData)
    {
        ArgumentNullException.ThrowIfNull(xData);
        return Fit(xData.Select(x => new[] { x }).ToArray(), yData, new[] { "x" });
    }

    public Result Fit(double[][] features, double[] targets, string[]? featureNames = null,
        CancellationToken cancellationToken = default)
    {
        ValidateData(features, targets, featureNames);
        _featureCount = features[0].Length;
        _featureNames = featureNames is null
            ? Enumerable.Range(0, _featureCount).Select(i => i == 0 ? "x" : $"x{i}").ToArray()
            : (string[])featureNames.Clone();
        _history.Clear();
        _archive.Clear();

        (int[] train, int[] validation) = SplitIndices(features.Length);
        double targetVariance = Variance(train.Select(i => targets[i]));
        var population = InitializePopulation(targets, train);
        ExpressionTree? globalBest = null;
        double bestValidation = double.PositiveInfinity;
        int staleGenerations = 0;
        var watch = Stopwatch.StartNew();

        for (int generation = 0; generation < _config.MaxGenerations; generation++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EvaluatePopulation(population, features, targets, train, validation, targetVariance);
            population.Sort(CompareIndividuals);

            if (_config.ConstantOptimizationInterval > 0 && generation > 0 &&
                generation % _config.ConstantOptimizationInterval == 0)
            {
                foreach (ExpressionTree candidate in population.Take(_config.ConstantOptimizationCandidates))
                    OptimizeConstants(candidate, features, targets, train);
                EvaluatePopulation(population.Take(_config.ConstantOptimizationCandidates), features, targets, train, validation, targetVariance);
                population.Sort(CompareIndividuals);
            }

            UpdateArchive(population);
            ExpressionTree best = SelectBestGeneralizing(population);
            double selectionLoss = FiniteSelectionLoss(best);
            double medianFitness = Median(population.Select(p => p.Fitness));
            _history.Add(new GenerationData(generation + 1, best.Mse, best.ValidationLoss,
                medianFitness, best.Size, ExpressionSimplifier.MaterializeLinearScaling(best).ToInfixString(_featureNames)));

            if (globalBest is null || IsBetter(best, globalBest)) globalBest = best.DeepClone(metadata: true);
            if (selectionLoss + 1e-14 < bestValidation)
            {
                bestValidation = selectionLoss;
                staleGenerations = 0;
            }
            else staleGenerations++;

            if (_config.EnableLogging && (generation == 0 || (generation + 1) % 10 == 0))
                Console.WriteLine($"Generation {generation + 1,4}: train MSE={best.Mse:G6}, validation={best.ValidationLoss:G6}, nodes={best.Size}");

            if (best.Mse <= _config.TargetError || staleGenerations >= _config.EarlyStoppingPatience) break;
            population = CreateNextPopulation(population);
        }

        watch.Stop();
        if (globalBest is null) throw new InvalidOperationException("No model was produced.");
        ExpressionTree model = ExpressionSimplifier.MaterializeLinearScaling(globalBest);
        double[] predictions = features.Select(model.Evaluate).ToArray();
        ModelMetrics metrics = ModelMetrics.Calculate(targets, predictions);
        var historySnapshot = _history.ToArray();
        var pareto = BuildParetoResult();
        var modelSnapshot = model.DeepClone();

        var result = new Result
        {
            BestFormula = model.ToInfixString(_featureNames),
            FunctionalFormula = model.ToFunctionalString(_featureNames),
            Fitness = globalBest.Fitness,
            MSE = metrics.Mse,
            RMSE = metrics.Rmse,
            MAE = metrics.Mae,
            R2 = metrics.R2,
            ValidationLoss = globalBest.ValidationLoss,
            Size = model.GetSize(),
            Generations = historySnapshot.Length,
            TrainingTime = watch.Elapsed,
            Function = _featureCount == 1
                ? x => modelSnapshot.Evaluate(x)
                : _ => throw new InvalidOperationException("Use MultiFunction or Predict for a multivariate model."),
            MultiFunction = row => modelSnapshot.Evaluate(row),
            FeatureNames = Array.AsReadOnly((string[])_featureNames.Clone()),
            ParetoFront = pareto,
            History = historySnapshot
        };

        if (_config.GenerateReport) ReportWriter.Write(_config.OutputPath, result, targets, predictions);
        return result;
    }

    public Result FitWithReport(double[] xData, double[] yData)
    {
        Config reportConfig = _config.Copy();
        reportConfig.GenerateReport = true;
        reportConfig.EnableLogging = true;
        return new SymbolicRegressor(reportConfig).Fit(xData, yData);
    }

    public Result FitWithReport(double[][] features, double[] targets, string[]? featureNames = null,
        CancellationToken cancellationToken = default)
    {
        Config reportConfig = _config.Copy();
        reportConfig.GenerateReport = true;
        reportConfig.EnableLogging = true;
        return new SymbolicRegressor(reportConfig).Fit(features, targets, featureNames, cancellationToken);
    }

    public Result AutoFit(double[] xData, double[] yData)
    {
        ArgumentNullException.ThrowIfNull(xData);
        return AutoFit(xData.Select(x => new[] { x }).ToArray(), yData, new[] { "x" });
    }

    public Result AutoFit(double[][] features, double[] targets, string[]? featureNames = null,
        CancellationToken cancellationToken = default)
    {
        ValidateData(features, targets, featureNames);
        Config tuned = _config.Copy();
        int dimensions = features[0].Length;
        int observations = features.Length;
        tuned.PopulationSize = Math.Clamp(300 + 80 * dimensions + observations / 2, 400, 2400);
        tuned.MaxGenerations = Math.Clamp(160 + 20 * dimensions, 180, 600);
        tuned.MaxDepth = dimensions > 8 ? 6 : 7;
        tuned.ParsimonyCoefficient = dimensions > 4 ? 0.002 : 0.001;
        if (observations < 10) tuned.ValidationFraction = 0;
        return new SymbolicRegressor(tuned).Fit(features, targets, featureNames, cancellationToken);
    }

    private List<ExpressionTree> InitializePopulation(double[] targets, int[] train)
    {
        var population = new List<ExpressionTree>(_config.PopulationSize);
        for (int feature = 0; feature < _featureCount && population.Count < _config.PopulationSize; feature++)
        {
            var variable = Variable(feature);
            population.Add(variable);
            population.Add(new ExpressionTree("*", variable.DeepClone(), variable.DeepClone()));
        }

        double mean = train.Select(i => targets[i]).Average();
        population.Add(new ExpressionTree(mean));
        for (int left = 0; left < _featureCount && population.Count < _config.PopulationSize; left++)
            for (int right = left + 1; right < _featureCount && population.Count < _config.PopulationSize; right++)
                population.Add(new ExpressionTree("*", Variable(left), Variable(right)));

        int index = 0;
        while (population.Count < _config.PopulationSize)
        {
            int depth = 2 + index % Math.Max(1, _config.MaxDepth - 1);
            bool full = (index / Math.Max(1, _config.MaxDepth - 1)) % 2 == 0;
            population.Add(BuildTree(depth, full));
            index++;
        }
        return population;
    }

    private ExpressionTree BuildTree(int depth, bool full = false)
    {
        if (depth <= 1 || (!full && depth > 2 && _rng.NextDouble() < 0.32)) return RandomTerminal();
        string function = _config.Functions[_rng.Next(_config.Functions.Length)];
        ExpressionTree left = BuildTree(depth - 1, full);
        return ExpressionTree.IsUnary(function)
            ? new ExpressionTree(function, left)
            : new ExpressionTree(function, left, BuildTree(depth - 1, full));
    }

    private ExpressionTree RandomTerminal()
    {
        if (_rng.NextDouble() < 0.68) return Variable(_rng.Next(_featureCount));
        double value = _config.ConstantMin + _rng.NextDouble() * (_config.ConstantMax - _config.ConstantMin);
        return new ExpressionTree(value);
    }

    private static ExpressionTree Variable(int index) => new(index == 0 ? "x" : $"x{index}");

    private void EvaluatePopulation(IEnumerable<ExpressionTree> population, double[][] x, double[] y,
        int[] train, int[] validation, double targetVariance)
    {
        if (_config.ParallelEvaluation)
            Parallel.ForEach(population, tree => EvaluateIndividual(tree, x, y, train, validation, targetVariance));
        else
            foreach (ExpressionTree tree in population) EvaluateIndividual(tree, x, y, train, validation, targetVariance);
    }

    private void EvaluateIndividual(ExpressionTree tree, double[][] x, double[] y,
        int[] train, int[] validation, double targetVariance)
    {
        double[] raw = new double[train.Length];
        bool valid = true;
        for (int i = 0; i < train.Length; i++)
        {
            raw[i] = tree.Evaluate(x[train[i]]);
            if (!double.IsFinite(raw[i]) || Math.Abs(raw[i]) >= 1e99) { valid = false; break; }
        }

        tree.Size = tree.GetSize();
        if (!valid)
        {
            tree.Fitness = tree.Mse = tree.ValidationLoss = double.PositiveInfinity;
            tree.CaseErrors = Enumerable.Repeat(double.PositiveInfinity, train.Length).ToArray();
            return;
        }

        (tree.LinearScale, tree.LinearOffset) = _config.EnableLinearScaling
            ? FitLinearScale(raw, train.Select(i => y[i]).ToArray())
            : (1.0, 0.0);

        var squared = new double[train.Length];
        tree.CaseErrors = new double[train.Length];
        for (int i = 0; i < train.Length; i++)
        {
            double prediction = tree.LinearScale * raw[i] + tree.LinearOffset;
            double residual = prediction - y[train[i]];
            squared[i] = residual * residual;
            tree.CaseErrors[i] = Math.Abs(residual);
        }

        tree.Mse = squared.Average();
        double trainingLoss = CalculateLoss(tree.CaseErrors);
        tree.ValidationLoss = validation.Length == 0
            ? trainingLoss
            : CalculateLoss(validation.Select(i => Math.Abs(tree.EvaluateModel(x[i]) - y[i])));
        double normalizedLoss = trainingLoss / Math.Max(targetVariance, 1e-12);
        tree.Fitness = normalizedLoss + _config.ParsimonyCoefficient * tree.Size;
        if (!double.IsFinite(tree.Fitness)) tree.Fitness = double.PositiveInfinity;
    }

    private double CalculateLoss(IEnumerable<double> absoluteErrors)
    {
        double[] errors = absoluteErrors as double[] ?? absoluteErrors.ToArray();
        return _config.Loss switch
        {
            LossFunction.MeanAbsoluteError => errors.Average(),
            LossFunction.Huber => errors.Average(e => e <= _config.HuberDelta
                ? 0.5 * e * e
                : _config.HuberDelta * (e - 0.5 * _config.HuberDelta)),
            _ => errors.Average(e => e * e)
        };
    }

    private static (double scale, double offset) FitLinearScale(double[] values, double[] targets)
    {
        double xMean = values.Average();
        double yMean = targets.Average();
        double covariance = 0, variance = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double centered = values[i] - xMean;
            covariance += centered * (targets[i] - yMean);
            variance += centered * centered;
        }
        if (variance <= 1e-20) return (0.0, yMean);
        double scale = Math.Clamp(covariance / variance, -1e6, 1e6);
        double offset = yMean - scale * xMean;
        return double.IsFinite(offset) ? (scale, offset) : (1.0, 0.0);
    }

    private List<ExpressionTree> CreateNextPopulation(List<ExpressionTree> population)
    {
        var next = new List<ExpressionTree>(_config.PopulationSize);
        int elites = Math.Max(1, (int)Math.Round(_config.PopulationSize * _config.SurvivalRate));
        for (int i = 0; i < elites; i++) next.Add(population[i].DeepClone());

        int immigrants = (int)Math.Round(_config.PopulationSize * _config.RandomImmigrantRate);
        for (int i = 0; i < immigrants && next.Count < _config.PopulationSize; i++)
            next.Add(BuildTree(_rng.Next(2, _config.MaxDepth + 1)));

        int rejected = 0;
        while (next.Count < _config.PopulationSize)
        {
            ExpressionTree parent = SelectParent(population);
            ExpressionTree child = _rng.NextDouble() < _config.CrossoverRate
                ? Crossover(parent, SelectParent(population))
                : parent.DeepClone();
            if (_rng.NextDouble() < _config.MutationRate) child = Mutate(child);
            child = ExpressionSimplifier.Simplify(child);
            if (child.GetDepth() <= _config.MaxDepth && child.GetSize() <= _config.MaxNodes)
            {
                next.Add(child);
                rejected = 0;
            }
            else if (++rejected > _config.PopulationSize)
            {
                next.Add(BuildTree(_rng.Next(2, Math.Min(4, _config.MaxDepth) + 1)));
                rejected = 0;
            }
        }
        return next;
    }

    private ExpressionTree SelectParent(List<ExpressionTree> population) =>
        _config.Selection == ParentSelection.EpsilonLexicase ? EpsilonLexicase(population) : Tournament(population);

    private ExpressionTree Tournament(List<ExpressionTree> population)
    {
        ExpressionTree? best = null;
        for (int i = 0; i < _config.TournamentSize; i++)
        {
            ExpressionTree candidate = population[_rng.Next(population.Count)];
            if (best is null || CompareIndividuals(candidate, best) < 0) best = candidate;
        }
        return best!;
    }

    private ExpressionTree EpsilonLexicase(List<ExpressionTree> population)
    {
        int poolSize = Math.Min(_config.LexicasePoolSize, population.Count);
        var survivors = new List<ExpressionTree>(poolSize);
        for (int i = 0; i < poolSize; i++) survivors.Add(population[_rng.Next(population.Count)]);
        if (survivors[0].CaseErrors.Length == 0) return Tournament(population);

        int[] cases = Enumerable.Range(0, survivors[0].CaseErrors.Length).OrderBy(_ => _rng.Next()).Take(32).ToArray();
        foreach (int caseIndex in cases)
        {
            double best = survivors.Min(s => s.CaseErrors[caseIndex]);
            double med = Median(survivors.Select(s => s.CaseErrors[caseIndex]));
            double mad = Median(survivors.Select(s => Math.Abs(s.CaseErrors[caseIndex] - med)));
            survivors = survivors.Where(s => s.CaseErrors[caseIndex] <= best + mad + 1e-12).ToList();
            if (survivors.Count <= 1) break;
        }
        return survivors[_rng.Next(survivors.Count)];
    }

    private ExpressionTree Crossover(ExpressionTree first, ExpressionTree second)
    {
        ExpressionTree child = first.DeepClone();
        List<string> firstPaths = GetPaths(child);
        List<string> secondPaths = GetPaths(second);
        string destination = firstPaths[_rng.Next(firstPaths.Count)];
        string source = secondPaths[_rng.Next(secondPaths.Count)];
        ExpressionTree replacement = GetAtPath(second, source).DeepClone();
        return ReplaceAtPath(child, destination, replacement);
    }

    private ExpressionTree Mutate(ExpressionTree tree)
    {
        double choice = _rng.NextDouble();
        if (choice < 0.50)
        {
            string path = RandomPath(tree);
            return ReplaceAtPath(tree, path, BuildTree(_rng.Next(1, _config.MaxMutationDepth + 1)));
        }
        if (choice < 0.72) return PointMutation(tree);
        if (choice < 0.86) return ConstantMutation(tree);
        return HoistMutation(tree);
    }

    private ExpressionTree PointMutation(ExpressionTree tree)
    {
        ExpressionTree result = tree.DeepClone();
        ExpressionTree node = GetAtPath(result, RandomPath(result));
        if (node.IsLeaf)
        {
            ExpressionTree replacement = RandomTerminal();
            node.Value = replacement.Value;
        }
        else
        {
            bool unary = node.Right is null;
            string[] compatible = _config.Functions.Where(f => ExpressionTree.IsUnary(f) == unary).ToArray();
            if (compatible.Length > 0) node.Value = compatible[_rng.Next(compatible.Length)];
        }
        return result;
    }

    private ExpressionTree ConstantMutation(ExpressionTree tree)
    {
        ExpressionTree result = tree.DeepClone();
        List<ExpressionTree> constants = GetPaths(result).Select(p => GetAtPath(result, p)).Where(n => n.IsConstant).ToList();
        if (constants.Count == 0) return PointMutation(result);
        ExpressionTree node = constants[_rng.Next(constants.Count)];
        double value = double.Parse(node.Value, CultureInfo.InvariantCulture);
        value += NextGaussian() * Math.Max(0.1, Math.Abs(value) * 0.2);
        node.Value = Math.Clamp(value, _config.ConstantMin * 10, _config.ConstantMax * 10).ToString("G12", CultureInfo.InvariantCulture);
        return result;
    }

    private ExpressionTree HoistMutation(ExpressionTree tree)
    {
        ExpressionTree result = tree.DeepClone();
        List<string> paths = GetPaths(result).Where(p => !GetAtPath(result, p).IsLeaf).ToList();
        if (paths.Count == 0) return result;
        string path = paths[_rng.Next(paths.Count)];
        ExpressionTree node = GetAtPath(result, path);
        ExpressionTree replacement = node.Right is null || _rng.Next(2) == 0 ? node.Left! : node.Right;
        return ReplaceAtPath(result, path, replacement.DeepClone());
    }

    private void OptimizeConstants(ExpressionTree tree, double[][] x, double[] y, int[] train)
    {
        List<ExpressionTree> constants = GetPaths(tree).Select(p => GetAtPath(tree, p)).Where(n => n.IsConstant).ToList();
        if (constants.Count == 0) return;
        double best = RawMse(tree, x, y, train);
        foreach (ExpressionTree constant in constants)
        {
            double original = double.Parse(constant.Value, CultureInfo.InvariantCulture);
            double step = Math.Max(0.25, Math.Abs(original) * 0.25);
            for (int pass = 0; pass < _config.ConstantOptimizationPasses; pass++)
            {
                double accepted = original;
                foreach (double candidate in new[] { original - step, original + step })
                {
                    constant.Value = candidate.ToString("G12", CultureInfo.InvariantCulture);
                    double score = RawMse(tree, x, y, train);
                    if (score < best) { best = score; accepted = candidate; }
                }
                original = accepted;
                constant.Value = original.ToString("G12", CultureInfo.InvariantCulture);
                step *= 0.5;
            }
        }
    }

    private static double RawMse(ExpressionTree tree, double[][] x, double[] y, int[] indices)
    {
        double sum = 0;
        foreach (int i in indices)
        {
            double error = tree.Evaluate(x[i]) - y[i];
            if (!double.IsFinite(error) || Math.Abs(error) > 1e50) return double.PositiveInfinity;
            sum += error * error;
        }
        return sum / indices.Length;
    }

    private void UpdateArchive(List<ExpressionTree> population)
    {
        var candidates = _archive.Concat(population.Take(80)).Select(t => t.DeepClone(metadata: true))
            .Where(t => double.IsFinite(t.ValidationLoss))
            .GroupBy(t => ExpressionSimplifier.MaterializeLinearScaling(t).ToFunctionalString())
            .Select(g => g.OrderBy(FiniteSelectionLoss).First()).ToList();
        _archive.Clear();
        foreach (ExpressionTree candidate in candidates)
        {
            bool dominated = candidates.Any(other => !ReferenceEquals(other, candidate) &&
                other.Size <= candidate.Size && FiniteSelectionLoss(other) <= FiniteSelectionLoss(candidate) &&
                (other.Size < candidate.Size || FiniteSelectionLoss(other) < FiniteSelectionLoss(candidate)));
            if (!dominated) _archive.Add(candidate);
        }
        _archive.Sort((a, b) => a.Size != b.Size ? a.Size.CompareTo(b.Size) : FiniteSelectionLoss(a).CompareTo(FiniteSelectionLoss(b)));
        if (_archive.Count > 40) _archive.RemoveRange(40, _archive.Count - 40);
    }

    private IReadOnlyList<ParetoSolution> BuildParetoResult() => _archive.Select(tree =>
    {
        ExpressionTree model = ExpressionSimplifier.MaterializeLinearScaling(tree);
        return new ParetoSolution(model.ToInfixString(_featureNames), model.ToFunctionalString(_featureNames),
            model.GetSize(), tree.Mse, tree.ValidationLoss);
    }).ToArray();

    private static ExpressionTree SelectBestGeneralizing(List<ExpressionTree> population) =>
        population.OrderBy(FiniteSelectionLoss).ThenBy(p => p.Size).ThenBy(p => p.Mse).First();

    private static bool IsBetter(ExpressionTree candidate, ExpressionTree incumbent)
    {
        double a = FiniteSelectionLoss(candidate), b = FiniteSelectionLoss(incumbent);
        return a < b - 1e-14 || (Math.Abs(a - b) <= 1e-14 && candidate.Size < incumbent.Size);
    }

    private static int CompareIndividuals(ExpressionTree a, ExpressionTree b)
    {
        int fitness = a.Fitness.CompareTo(b.Fitness);
        return fitness != 0 ? fitness : a.Size.CompareTo(b.Size);
    }

    private static double FiniteSelectionLoss(ExpressionTree tree) =>
        double.IsFinite(tree.ValidationLoss) ? tree.ValidationLoss : tree.Mse;

    private (int[] train, int[] validation) SplitIndices(int count)
    {
        int[] all = Enumerable.Range(0, count).OrderBy(_ => _rng.Next()).ToArray();
        int validationCount = count >= 8 ? (int)Math.Round(count * _config.ValidationFraction) : 0;
        validationCount = Math.Clamp(validationCount, 0, Math.Max(0, count - 3));
        return (all.Skip(validationCount).ToArray(), all.Take(validationCount).ToArray());
    }

    private static List<string> GetPaths(ExpressionTree root)
    {
        var paths = new List<string>();
        Walk(root, "", paths);
        return paths;

        static void Walk(ExpressionTree node, string path, List<string> output)
        {
            output.Add(path);
            if (node.Left is not null) Walk(node.Left, path + "L", output);
            if (node.Right is not null) Walk(node.Right, path + "R", output);
        }
    }

    private string RandomPath(ExpressionTree tree)
    {
        List<string> paths = GetPaths(tree);
        return paths[_rng.Next(paths.Count)];
    }

    private static ExpressionTree GetAtPath(ExpressionTree root, string path)
    {
        ExpressionTree current = root;
        foreach (char step in path) current = step == 'L' ? current.Left! : current.Right!;
        return current;
    }

    private static ExpressionTree ReplaceAtPath(ExpressionTree root, string path, ExpressionTree replacement)
    {
        if (path.Length == 0) return replacement;
        ExpressionTree parent = GetAtPath(root, path[..^1]);
        if (path[^1] == 'L') parent.Left = replacement; else parent.Right = replacement;
        return root;
    }

    private double NextGaussian()
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double Variance(IEnumerable<double> values)
    {
        double[] data = values.ToArray();
        double mean = data.Average();
        return data.Average(value => (value - mean) * (value - mean));
    }

    private static double Median(IEnumerable<double> values)
    {
        double[] sorted = values.Where(double.IsFinite).OrderBy(x => x).ToArray();
        if (sorted.Length == 0) return double.PositiveInfinity;
        int middle = sorted.Length / 2;
        return sorted.Length % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
    }

    private static void ValidateData(double[][] features, double[] targets, string[]? names)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(targets);
        if (features.Length != targets.Length) throw new ArgumentException("Features and targets must have the same number of rows.");
        if (features.Length < 3) throw new ArgumentException("At least three observations are required.", nameof(features));
        if (features[0] is null || features[0].Length == 0) throw new ArgumentException("At least one feature is required.", nameof(features));
        int width = features[0].Length;
        for (int row = 0; row < features.Length; row++)
        {
            if (features[row] is null || features[row].Length != width) throw new ArgumentException("All feature rows must have the same length.", nameof(features));
            if (features[row].Any(v => !double.IsFinite(v))) throw new ArgumentException($"Feature row {row} contains NaN or infinity.", nameof(features));
            if (!double.IsFinite(targets[row])) throw new ArgumentException($"Target {row} contains NaN or infinity.", nameof(targets));
        }
        if (names is not null)
        {
            if (names.Length != width) throw new ArgumentException("Feature name count must match the feature count.", nameof(names));
            if (names.Any(string.IsNullOrWhiteSpace) || names.Distinct(StringComparer.Ordinal).Count() != names.Length)
                throw new ArgumentException("Feature names must be non-empty and unique.", nameof(names));
        }
    }
}
