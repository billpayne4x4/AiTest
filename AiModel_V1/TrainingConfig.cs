namespace AiModel_V1;

/// <summary>
/// All depth-scaling training techniques in one config. Toggle from the AI menu.
/// </summary>
public class TrainingConfig
{
    public bool UseResidual = false; // only helps at depth; off for shallow defaults
    public bool UseLayerNorm = true;
    // 0=tanh 1=relu 2=gelu
    public int Activation = 2;
    public bool UseAdam = true;
    // 0=xavier 1=he 2=orthogonal
    public int Init = 1;
    public float GradClip = 1.0f;
    public bool UsePpo = true;
    public float EntropyBonus = 0.01f;

    public TrainingConfig Clone() => (TrainingConfig)MemberwiseClone();
}
