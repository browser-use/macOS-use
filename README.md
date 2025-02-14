<picture>
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/macos-use.png"  width="full">
</picture>

<br/>

[![GitHub stars](https://img.shields.io/github/stars/browser-use/macOS-use?style=social)](https://github.com/browser-use/macOS-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Twitter Follow](https://img.shields.io/twitter/follow/OfirOzeri?style=social)](https://x.com/OfirOzeri)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00)

<div align="center">
  <h2>Tell your MacBook what to do, and it's done—across ANY app.</h2>
  Created by <a href="https://x.com/OfirOzeri">Ofir Ozeri </a>♥️ migrated in collaboration with <a href="https://x.com/mamagnus00">Magnus</a> and <a href="https://x.com/gregpr07">Gregor</a><br>
</div>
<br>

macOS-use enables AI agents to interact with your Macbook [see it in action!](#demos)

# Quick start

⚠️ Important: Review the [Warning](#warning) section before proceeding. <br>

### With pip:

```bash
pip install mlx-use
```

### From github

Clone first
<br>

```bash
git clone https://github.com/browser-use/macOS-use.git && cd macOS-use
```

Don't forget API key <br>Supported providers: [OAI](https://platform.openai.com/docs/quickstart), [Anthropic](https://docs.anthropic.com/en/api/admin-api/apikeys/get-api-key), [Gemini](https://ai.google.dev/gemini-api/docs/api-key) or [Tongyi Qwen](https://www.alibabacloud.com/en/product/modelstudio) (deepseek R1 coming soon!)

<br> At the moment, macOS-use works best with OAI or Anthropic API, tho Gemini is free. it works great two, just not as reliably. <br>

```bash
cp .env.example .env
```

```bash
open ./.env
```

We recommend using macOS-use with uv environment
<br>

```bash
brew install uv && uv venv && ./.venv/bin/activate
```

Install locally and you're good to go! try the first exmaple!
<br>

```bash
uv pip install --editable . && python examples/try.py

```

Try prompting it with

```bash
open the calculator app
```

# Demos

<h3> Click the GIF for the full video! </h3>

[prompt](https://github.com/browser-use/macOS-use/blob/main/examples/calculate.py): Calculate how much is 5 X 4 and return the result, then call done.

```bash
python examples/calculate.py

```

<br>

[![calc-5-times-4](https://github.com/browser-use/macOS-use/blob/main/static/calc-5-X-4.gif "Click the GIF for full video!")](https://x.com/OfirOzeri/status/1883110905665433681)

<br/>

[prompt](https://github.com/browser-use/macOS-use/blob/main/examples/login_to_auth0.py): Go to auth0.com, sign in with google auth, choose ofiroz91 gmail account, login to the website and call done when you finish.

```bash
python examples/login_to_auth0.py
```

 <br>

[![login-to-auth0](https://github.com/browser-use/macOS-use/blob/main/static/login-to-auth0.gif "Click for full video")](https://x.com/OfirOzeri/status/1883455599423434966)

<br/>

[prompt](https://github.com/browser-use/macOS-use/blob/main/examples/check_time_online.py): Can you check what hour is Shabbat in israel today? call done when you finish.

```bash
python examples/check_time_online.py
```

<br>

[![check-time-online](https://github.com/browser-use/macOS-use/blob/main/static/check-time-online.gif "Click for full video")](https://x.com/OfirOzeri/status/1883109604416278552)

<br>

# Our Vision:

TLDR: Tell every Apple device what to do, and see it done. on EVERY APP.
<br><br>
This project aimes to build the AI agent for the MLX by Apple framework that would allow the agent to perform any action on any Apple device. Our final goal is a open source that anyone can clone, powered by the [mlx](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) to run local private infrence at zero cost.

## Roadmap goals:

1. Support MacBooks at SOTA reliability

- [ ] Refine the Agent prompting.
- [ ] Release the first working version to pypi.
- [ ] Improve self-correction.
- [x] Adding ability to check which apps the machine has installed.
- [x] Add feature to allow the agent to check existing apps if failing, e.g. calendar app actual name is iCal.
- [ ] Add action for the agent to ask input from the user.
- [ ] Test Test Test! and let us know what and how to improve!
- [ ] Make task cheaper and more efficient.

2. Support local inference with small fine tuned model.

- [ ] Add support for inference with local models using mlx and mlx-vlm.
- [ ] Fine tune a small model that every device can run inference with.
- [ ] SOTA reliability.

3. Support iPhone/iPad

<br>

# WARNING

This project is still under development and user discretion is advised!
macOS-use can and will use your do [login](#demos), use private credentials, [auth services](https://github.com/browser-use/macOS-use/blob/main/examples/login_to_auth0.py) or stored passwords to complete its task, launch and interact WITH EVERY APP and UI component in your MacBook and restrictions to the model are still under active development! It is not recommended to operate it unsupervised YET
macOS-use WILL NOT STOP at captcha or any other forms of bot identifications, so once again, user discretion is advised.

## Disclaimer:

As this is an early stage release, You might experience varying success rates depending on task prompt, we're actively working on improvements. <br> talk me on [X/Twitter](https://x.com/OfirOzeri) or contact me on [Discord](https://link.browser-use.com/discord), your input is crucial and highly valuable!<br>

# Contributing

We are a new project and would love contributors! Feel free to PR, open issues for bugs or feature requests.

# Thanks

I would like to extend my heartfelt thanks to [![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07) and [![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00) for their incredible work in developing Browser Use. Their dedication and expertise have been invaluable, especially in helping with the migration process and I couldn't have done it without them!
