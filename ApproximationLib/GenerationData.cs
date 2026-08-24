namespace ApproximationLib;

public sealed record GenerationData(
    int GenerationNumber,
    double BestTrainingLoss,
    double BestValidationLoss,
    double MedianFitness,
    int BestSize,
    string BestFormula);
