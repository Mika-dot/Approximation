using System.Globalization;

namespace ApproximationLib;

internal static class ExpressionSimplifier
{
    public static ExpressionTree Simplify(ExpressionTree tree)
    {
        var node = new ExpressionTree(tree.Value,
            tree.Left is null ? null : Simplify(tree.Left),
            tree.Right is null ? null : Simplify(tree.Right));

        if (node.IsLeaf) return node;
        if (node.Left?.IsConstant == true && (node.Right is null || node.Right.IsConstant))
        {
            double value = node.Evaluate(Array.Empty<double>());
            if (double.IsFinite(value) && Math.Abs(value) < 1e50) return Constant(value);
        }

        if (node.Right is not null)
        {
            if (node.Value == "+")
            {
                if (IsZero(node.Left)) return node.Right;
                if (IsZero(node.Right)) return node.Left!;
            }
            else if (node.Value == "-")
            {
                if (IsZero(node.Right)) return node.Left!;
                if (node.Left!.ToFunctionalString() == node.Right.ToFunctionalString()) return Constant(0);
            }
            else if (node.Value == "*")
            {
                if (IsZero(node.Left) || IsZero(node.Right)) return Constant(0);
                if (IsOne(node.Left)) return node.Right;
                if (IsOne(node.Right)) return node.Left!;
            }
            else if (node.Value == "protectedDiv")
            {
                if (IsZero(node.Left)) return Constant(0);
                if (IsOne(node.Right)) return node.Left!;
            }
        }
        return node;
    }

    public static ExpressionTree MaterializeLinearScaling(ExpressionTree tree)
    {
        ExpressionTree result = tree.DeepClone();
        if (Math.Abs(tree.LinearScale - 1.0) > 1e-12)
            result = new ExpressionTree("*", Constant(tree.LinearScale), result);
        if (Math.Abs(tree.LinearOffset) > 1e-12)
            result = new ExpressionTree("+", result, Constant(tree.LinearOffset));
        return Simplify(result);
    }

    private static ExpressionTree Constant(double value) =>
        new(value.ToString("G12", CultureInfo.InvariantCulture));

    private static bool IsZero(ExpressionTree? node) => ConstantEquals(node, 0);
    private static bool IsOne(ExpressionTree? node) => ConstantEquals(node, 1);

    private static bool ConstantEquals(ExpressionTree? node, double expected) =>
        node?.IsConstant == true &&
        double.TryParse(node.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) &&
        Math.Abs(value - expected) <= 1e-12;
}
