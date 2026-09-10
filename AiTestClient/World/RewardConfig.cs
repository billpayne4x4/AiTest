namespace AiTestClient.World;

/// <summary>Tunable reward shaping, editable live from the rewards menu (W).</summary>
public class RewardConfig
{
    public float Speed = 2.5f;      // reward per speed01
    public float Centering = 0.5f;  // centerline bonus * speed
    public float Alignment = 1.5f;  // facing track direction * speed
    public float Progress = 1500f;  // per unit of lap progress
    public float WrongWay = 1.5f;   // penalty when facing backwards
    public float Slide = 3f;        // penalty per slide01
    public float Spin = 5f;         // flat penalty while spinning
    public float Understeer = 1f;   // flat penalty while plowing
    public float SteerEffort = 0.08f;
    public float OffTrack = 12f;    // flat penalty while off road

    public RewardConfig Clone() => (RewardConfig)MemberwiseClone();
}
