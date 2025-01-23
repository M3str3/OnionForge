# OnionForge
OnionForge generates custom .onion addresses for Tor hidden services using brute force. Specify a prefix or regex pattern to find matching .onion addresses. Includes a simulation mode to estimate the time needed to find a match.

<p align="center">
<img src="https://github.com/M3str3/OnionForge/assets/62236987/dcc0f4b5-a6a4-4fba-b91a-7f5f48439919" width=50% height=50% style="display: block; margin: 0 auto">
</p>

## Usage

### Basic Usage

To run the program, use the following command:

```sh
cargo run --release <prefix> [--simulate] [--no-stop]
```

- `<prefix>`: The prefix you want the .onion address to start with. If it starts with r/, it is treated as a regex pattern.
- `--simulate`: Run the program in simulation mode for 10 seconds to estimate generation time.
- `--no-stop`: Continue generating and saving keys even after finding a match.
- `--no-header`: Do not include the header required by Tor service (`ed25519v1-secret: type0`).

## Examples
### Generate an Address with a Specific Prefix
`cargo run --release myprefix`

### Generate an Address with a Regex Pattern
`cargo run --release /r^mypattern.*`
### Run in Simulation Mode
`cargo run --release myprefix --simulate`

No simulation mode for regex patterns
### Generate and Save Keys Continuously

`cargo run --release myprefix --no-stop`

## Dependencies
- `torut` crate for Tor Onion service key generation.
- `regex` crate for regex pattern matching.
- `num_cpus` crate for detecting the number of CPUs.
