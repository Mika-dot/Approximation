using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApproximationLib
{
    // ==================== НАСТРОЙКИ ПО УМОЛЧАНИЮ ====================
    public class Config
    {
        public int PopulationSize { get; set; } = 2000;
        public int MaxGenerations { get; set; } = 500;
        public double SurvivalRate { get; set; } = 0.05;
        public double CrossoverRate { get; set; } = 0.8;
        public double MutationRate { get; set; } = 0.9;
        public int MaxDepth { get; set; } = 4;
        public int MaxMutationDepth { get; set; } = 4;
        public double ParsimonyCoefficient { get; set; } = 0.01;
        public int TournamentSize { get; set; } = 3;
        public int TopSolutionsToTrack { get; set; } = 5;
        public bool EnableLogging { get; set; } = false;
        public string OutputPath { get; set; } = "report.html";

        // Функции для использования
        public string[] Functions { get; set; } = {
                "+", "-", "*", "protectedDiv", "pow",
                "sin", "cos", "tan",
                "asin", "acos", "atan",
                "sinh", "cosh", "tanh",
                "log", "log10", "sqrt", "abs", "exp",
                "min", "max"
            };

        public string[] Terminals { get; set; } = { "x" };
    }
}
