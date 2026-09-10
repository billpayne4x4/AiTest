# 🏎️ AiTest — Neural Network Self-Driving Car

A from-scratch self-driving car experiment in **C# / .NET 10**: a tiny feed-forward neural network learns to drive a spline track using vision rays + policy-gradient reinforcement learning. Rendering is raw **SDL2 + OpenGL** (no game engine).

![Screenshot 1](Images/image1.png)
![Screenshot 2](Images/image2.png)
![Screenshot 3](Images/image3.png)
![Screenshot 4](Images/image4.png)
![Screenshot 5](Images/image5.png)

## ✨ Features

- 🧠 **From-scratch neural net** — feed-forward + backprop, no ML library
- 🔬 **Depth-scaling techniques** — residual skips (1/√2 scaled), LayerNorm, ReLU/GELU/tanh switch, Adam, He/Xavier/orthogonal init, gradient clipping, PPO-clipped advantage + entropy bonus, depth-scaled LR, weight decay, activation clamps
- 🚗 **Policy-gradient brain (`CarBrain`)** — vision rays → steer/throttle, reward-shaped learning with exploration noise + PPO critic
- 👁️ **Modular vision rays** — `Default5` fan or `Wide7` fan, configurable angles / range / step
- 🛣️ **Spline track** — Catmull-Rom closed loop, editable control points, spatial-hash ray casting
- 🎮 **3D renderer** — lit ground, road + curbs + dashed centerline, car mesh, vision-ray debug lines
- 📷 **3 camera modes** — Orbit / Chase / Free (keys `1/2/3`)
- ⚡ **Speed modes** — Normal / Step-through (`SPACE`) / SuperFast training
- 🖥️ **Live HUD** — 5×7 bitmap font, stats readout, brain visualization (nodes light up, weights colored by sign/magnitude)
- 🛠️ **In-app track editor** — drag control points (`T` mode), rebuild spline live
- 🔄 **Retraining menu** — change hidden layers / nodes + all 7+ training techniques at runtime (`R` / `M` / `C`, or click **[AI SETUP]**)
- ⚙️ **AI Setup menu** — toggle residual, LayerNorm, activation, Adam, init scheme, grad clip, PPO (`↑/↓` select, `←/→` change, `Enter` apply + retrain)
- 💻 **Headless mode** — `dotnet run -- --headless [steps]` trains with no window/GPU

## 🚀 Getting Started

```bash
dotnet build AiTest.sln
dotnet run --project AiTestClient
# headless training:
dotnet run --project AiTestClient -- --headless 6000
```

Requires .NET 10 SDK + SDL2 native libs.

## 🎹 Controls

| Key | Action |
|-----|--------|
| `1/2/3` | Orbit / Chase / Free camera |
| `N/F/S` | Normal / Step / SuperFast speed |
| `SPACE` | Advance one step (in Step mode) |
| `R` | Retrain (new brain) |
| `M` / `C` | AI config menu (layers/nodes + training techniques; or click [AI SETUP]) |
| `T` | Track editor |
| `B` | Toggle brain overlay |
| `ESC` | Quit |

## 📁 Project Structure

- `AiModel_V1/` — 🧠 `NeuralNetwork.cs`, `CarBrain.cs`, `VisionRays.cs`, `TrainingConfig.cs` (world-agnostic AI)
- `AiTestClient/` — 🎮 `Program.cs` (game loop), `Renderer.cs`, `Hud.cs`, `GameWindow.cs`, `World/` (`Car.cs`, `Track.cs`, `Simulation.cs`), `Native/` (SDL/OpenGL bindings)
- `Images/` — 📸 screenshots
