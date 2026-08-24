using System.Globalization;

namespace ApproximationLib;

/// <summary>A mathematical expression represented as a tree.</summary>
public sealed class ExpressionTree
{
    private const double Epsilon = 1e-12;
    private const double Limit = 1e100;

    public static readonly IReadOnlySet<string> SupportedFunctions = new HashSet<string>(StringComparer.Ordinal)
    {
        "+", "-", "*", "protectedDiv", "pow", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "log", "log10", "sqrt", "abs", "exp", "min", "max"
    };

    internal static readonly IReadOnlySet<string> UnaryFunctions = new HashSet<string>(StringComparer.Ordinal)
    {
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "log", "log10", "sqrt", "abs", "exp"
    };

    public string Value { get; set; }
    public ExpressionTree? Left { get; set; }
    public ExpressionTree? Right { get; set; }
    public double Fitness { get; internal set; } = double.PositiveInfinity;
    public double Mse { get; internal set; } = double.PositiveInfinity;
    public double TrainingLoss { get; internal set; } = double.PositiveInfinity;
    public double ValidationLoss { get; internal set; } = double.PositiveInfinity;
    public int Size { get; internal set; }
    internal double[] CaseErrors { get; set; } = Array.Empty<double>();
    internal double LinearScale { get; set; } = 1.0;
    internal double LinearOffset { get; set; }

    public ExpressionTree(string value, ExpressionTree? left = null, ExpressionTree? right = null)
    {
        Value = value ?? throw new ArgumentNullException(nameof(value));
        Left = left;
        Right = right;
    }

    public ExpressionTree(double constant)
        : this(constant.ToString("R", CultureInfo.InvariantCulture)) { }

    public bool IsLeaf => Left is null && Right is null;
    public bool IsConstant => IsLeaf && double.TryParse(Value, NumberStyles.Float, CultureInfo.InvariantCulture, out _);

    public double Evaluate(double x) => Evaluate(new[] { x });

    public double Evaluate(IReadOnlyList<double> variables)
    {
        double value = EvaluateCore(variables);
        return FiniteOrPenalty(value);
    }

    internal double EvaluateModel(IReadOnlyList<double> variables) =>
        FiniteOrPenalty(LinearScale * EvaluateCore(variables) + LinearOffset);

    private double EvaluateCore(IReadOnlyList<double> variables)
    {
        if (TryVariableIndex(Value, out int index))
        {
            if ((uint)index >= (uint)variables.Count) throw new ArgumentException($"Variable x{index} is missing from the input row.");
            return variables[index];
        }

        if (double.TryParse(Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double constant)) return constant;

        if (Left is null) return Limit;
        double a = Left.EvaluateCore(variables);

        if (Right is null)
        {
            return FiniteOrPenalty(Value switch
            {
                "sin" => Math.Sin(a),
                "cos" => Math.Cos(a),
                "tan" => Math.Abs(Math.Cos(a)) < 1e-9 ? 0.0 : Math.Tan(a),
                "asin" => Math.Asin(Math.Clamp(a, -1.0, 1.0)),
                "acos" => Math.Acos(Math.Clamp(a, -1.0, 1.0)),
                "atan" => Math.Atan(a),
                "sinh" => Math.Sinh(Math.Clamp(a, -30.0, 30.0)),
                "cosh" => Math.Cosh(Math.Clamp(a, -30.0, 30.0)),
                "tanh" => Math.Tanh(a),
                "log" => Math.Log(Math.Abs(a) + Epsilon),
                "log10" => Math.Log10(Math.Abs(a) + Epsilon),
                "sqrt" => Math.Sqrt(Math.Abs(a)),
                "abs" => Math.Abs(a),
                "exp" => Math.Exp(Math.Clamp(a, -60.0, 60.0)),
                _ => Limit
            });
        }

        double b = Right.EvaluateCore(variables);
        return FiniteOrPenalty(Value switch
        {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            "protectedDiv" => Math.Abs(b) < Epsilon ? a : a / b,
            "pow" => SafePow(a, b),
            "min" => Math.Min(a, b),
            "max" => Math.Max(a, b),
            _ => Limit
        });
    }

    public int GetSize() => 1 + (Left?.GetSize() ?? 0) + (Right?.GetSize() ?? 0);

    public int GetDepth() => IsLeaf ? 1 : 1 + Math.Max(Left?.GetDepth() ?? 0, Right?.GetDepth() ?? 0);

    public string ToFunctionalString(IReadOnlyList<string>? featureNames = null)
    {
        if (IsLeaf) return DisplayLeaf(featureNames);
        if (Right is null) return $"{Value}({Left!.ToFunctionalString(featureNames)})";
        string op = Value switch { "+" => "add", "-" => "sub", "*" => "mul", "protectedDiv" => "div", _ => Value };
        return $"{op}({Left!.ToFunctionalString(featureNames)}, {Right.ToFunctionalString(featureNames)})";
    }

    public string ToInfixString(IReadOnlyList<string>? featureNames = null)
    {
        if (IsLeaf) return DisplayLeaf(featureNames);
        if (Right is null) return $"{Value}({Left!.ToInfixString(featureNames)})";
        string left = Left!.ToInfixString(featureNames);
        string right = Right.ToInfixString(featureNames);
        return Value switch
        {
            "+" or "-" or "*" => $"({left} {Value} {right})",
            "protectedDiv" => $"({left} / {right})",
            "pow" => $"pow({left}, {right})",
            _ => $"{Value}({left}, {right})"
        };
    }

    public override string ToString() => ToFunctionalString();

    internal ExpressionTree DeepClone(bool metadata = false)
    {
        var clone = new ExpressionTree(Value, Left?.DeepClone(metadata), Right?.DeepClone(metadata));
        if (metadata)
        {
            clone.Fitness = Fitness;
            clone.Mse = Mse;
            clone.TrainingLoss = TrainingLoss;
            clone.ValidationLoss = ValidationLoss;
            clone.Size = Size;
            clone.CaseErrors = (double[])CaseErrors.Clone();
            clone.LinearScale = LinearScale;
            clone.LinearOffset = LinearOffset;
        }
        return clone;
    }

    internal static bool IsUnary(string value) => UnaryFunctions.Contains(value);

    internal static bool TryVariableIndex(string value, out int index)
    {
        if (value == "x") { index = 0; return true; }
        return value.Length > 1 && value[0] == 'x' && int.TryParse(value.AsSpan(1), NumberStyles.None, CultureInfo.InvariantCulture, out index);
    }

    private string DisplayLeaf(IReadOnlyList<string>? featureNames)
    {
        if (!TryVariableIndex(Value, out int index)) return Value;
        if (featureNames is not null && index < featureNames.Count) return featureNames[index];
        return index == 0 ? "x" : $"x{index}";
    }

    private static double SafePow(double a, double b)
    {
        b = Math.Clamp(b, -12.0, 12.0);
        if (a < 0 && Math.Abs(b - Math.Round(b)) > 1e-9) return 0.0;
        return Math.Pow(a, b);
    }

    private static double FiniteOrPenalty(double value)
    {
        if (!double.IsFinite(value)) return Math.CopySign(Limit, value);
        return Math.Clamp(value, -Limit, Limit);
    }
}
