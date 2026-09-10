namespace AiTestClient.World;

/// <summary>Tunable car-physics parameters, editable live from the physics menu (E).</summary>
public class PhysicsConfig
{
    public bool Realistic = true;   // off = simple arcade (no slide/spin)
    public float Grip = 26f;        // base lateral grip (higher = more planted)
    public float Downforce = 0.35f; // extra grip per m/s of speed
    public float Steering = 0.55f;  // max wheel angle rad (higher = twitchier)
    public float Oversteer = 1f;    // drift gain past the limit (0 = never slides)
    public float SpinThreshold = 11f;
    public bool Understeer = true;  // cap yaw past the grip limit (plow wide)

    public PhysicsConfig Clone() => (PhysicsConfig)MemberwiseClone();
}
