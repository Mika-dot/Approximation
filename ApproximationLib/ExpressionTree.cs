using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApproximationLib
{
    public class ExpressionTree
    {
        public string Value;
        public ExpressionTree Left;
        public ExpressionTree Right;
        public double Fitness;
        public double Mse;
        public int Size;

        public ExpressionTree(string value, ExpressionTree left = null, ExpressionTree right = null)
        {
            Value = value;
            Left = left;
            Right = right;
        }

        public ExpressionTree(double constant) : this(constant.ToString("F4")) { }

        public bool IsLeaf => Left == null && Right == null;

        public double Evaluate(double x)
        {
            if (Value == "x")
                return x;

            if (double.TryParse(Value, out double num))
                return num;

            // Унарные функции
            if (Left != null && Right == null)
            {
                double arg = Left.Evaluate(x);
                switch (Value)
                {
                    case "sin": return Math.Sin(arg);
                    case "cos": return Math.Cos(arg);
                    case "tan": return SafeTan(arg);
                    case "asin": return SafeAsin(arg);
                    case "acos": return SafeAcos(arg);
                    case "atan": return Math.Atan(arg);
                    case "sinh": return Math.Sinh(arg);
                    case "cosh": return Math.Cosh(arg);
                    case "tanh": return Math.Tanh(arg);
                    case "log": return SafeLog(arg);
                    case "log10": return SafeLog10(arg);
                    case "sqrt": return SafeSqrt(arg);
                    case "abs": return Math.Abs(arg);
                    case "exp": return Math.Exp(arg);
                }
            }

            // Бинарные функции
            if (Left != null && Right != null)
            {
                double leftVal = Left.Evaluate(x);
                double rightVal = Right.Evaluate(x);

                switch (Value)
                {
                    case "+": return leftVal + rightVal;
                    case "-": return leftVal - rightVal;
                    case "*": return leftVal * rightVal;
                    case "protectedDiv": return Math.Abs(rightVal) < 1e-9 ? 1.0 : leftVal / rightVal;
                    case "pow": return SafePow(leftVal, rightVal);
                    case "min": return Math.Min(leftVal, rightVal);
                    case "max": return Math.Max(leftVal, rightVal);
                }
            }

            return 0.0;
        }

        private double SafeTan(double x)
        {
            double cos = Math.Cos(x);
            return Math.Abs(cos) < 1e-9 ? 0.0 : Math.Sin(x) / cos;
        }

        private double SafeAsin(double x)
        {
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            return Math.Asin(x);
        }

        private double SafeAcos(double x)
        {
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            return Math.Acos(x);
        }

        private double SafeLog(double x)
        {
            return x <= 0 ? -10.0 : Math.Log(x);
        }

        private double SafeLog10(double x)
        {
            return x <= 0 ? -10.0 : Math.Log10(x);
        }

        private double SafeSqrt(double x)
        {
            return x < 0 ? 0.0 : Math.Sqrt(x);
        }

        private double SafePow(double a, double b)
        {
            try
            {
                if (a < 0 && Math.Abs(b - Math.Round(b)) > 1e-9)
                    return 0.0;
                double result = Math.Pow(a, b);
                return double.IsNaN(result) || double.IsInfinity(result) ? 0.0 : result;
            }
            catch
            {
                return 0.0;
            }
        }

        public int GetSize()
        {
            return 1 + (Left?.GetSize() ?? 0) + (Right?.GetSize() ?? 0);
        }

        public int GetDepth()
        {
            if (IsLeaf) return 1;
            int leftDepth = Left?.GetDepth() ?? 0;
            int rightDepth = Right?.GetDepth() ?? 0;
            return 1 + Math.Max(leftDepth, rightDepth);
        }

        public override string ToString()
        {
            if (IsLeaf)
                return Value == "x" ? "x" : Value;

            if (Right == null)
                return $"{Value}({Left})";

            string op = Value switch
            {
                "+" => "add",
                "-" => "sub",
                "*" => "mul",
                "protectedDiv" => "div",
                _ => Value
            };

            return $"{op}({Left}, {Right})";
        }
    }
}
