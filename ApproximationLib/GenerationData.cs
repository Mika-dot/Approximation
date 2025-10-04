using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApproximationLib
{
    public class GenerationData
    {
        public int GenerationNumber { get; set; }
        public List<ExpressionTree> Solutions { get; set; }

        public GenerationData(int generationNumber, List<ExpressionTree> solutions)
        {
            GenerationNumber = generationNumber;
            Solutions = solutions.Select(s => CloneTreeWithFitness(s)).ToList();
        }

        private ExpressionTree CloneTreeWithFitness(ExpressionTree tree)
        {
            var clone = Clone(tree);
            clone.Fitness = tree.Fitness;
            clone.Mse = tree.Mse;
            clone.Size = tree.Size;
            return clone;
        }

        private ExpressionTree Clone(ExpressionTree node)
        {
            if (node == null) return null;
            return new ExpressionTree(
                node.Value,
                Clone(node.Left),
                Clone(node.Right)
            );
        }
    }
}
