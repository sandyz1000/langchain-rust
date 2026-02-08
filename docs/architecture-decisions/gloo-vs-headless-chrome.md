# Architecture Decision: Why gloo is NOT used in browser_use Tool

## Context

The `browser_use` tool in `langchain-ai-rust` provides browser automation capabilities for LLM agents. During development, there was a question about whether we could use the `gloo` API instead of `headless_chrome`.

## Decision

**We use `headless_chrome` and NOT `gloo`** for the following reasons:

## Analysis

### What is gloo?

[gloo](https://gloo-rs.web.app/) is a modular toolkit of Rust crates for interacting with Web APIs when compiling to WebAssembly (WASM). It provides:
- DOM manipulation
- Event handling
- Browser storage access
- Timers and intervals
- Fetch API wrappers
- And more...

**Key Point**: gloo is designed for Rust code that runs *inside* a web browser as WASM.

### What is headless_chrome?

[headless_chrome](https://github.com/rust-headless-chrome/rust-headless-chrome) is a Rust library for controlling Chrome/Chromium browsers programmatically. It:
- Runs on the server/native environment
- Launches and controls an external browser process
- Uses the Chrome DevTools Protocol
- Can navigate, click, type, and scrape web content

**Key Point**: headless_chrome is designed for native code that *controls* a browser from the outside.

### Comparison

| Aspect | headless_chrome (✓ Current) | gloo (✗ Not Applicable) |
|--------|---------------------------|------------------------|
| **Execution Environment** | Native (server, desktop) | WASM (inside browser) |
| **Target Compilation** | `x86_64-unknown-linux-gnu`, etc. | `wasm32-unknown-unknown` |
| **Use Case** | Backend agents automating browsers | Frontend apps using browser APIs |
| **Browser Requirement** | Launches Chrome/Chromium process | Runs in browser context |
| **Network Access** | Full server-side networking | Browser sandbox restrictions |
| **File System** | Full OS file system access | Limited to browser storage APIs |

### Why gloo is NOT Suitable

1. **Incompatible Execution Models**
   - `langchain-ai-rust` is a **server-side LLM framework**
   - It supports backends like OpenAI, Claude, Bedrock, Ollama
   - All vector stores (PostgreSQL, Qdrant, etc.) are server-side
   - No WASM target is configured or intended

2. **Different Problem Domains**
   - `browser_use` needs to **control** a browser (navigate to arbitrary URLs, interact with pages)
   - `gloo` provides APIs for code **running inside** a browser page
   - These are fundamentally different use cases

3. **WASM Not Used Anywhere**
   - No `wasm-bindgen`, `web-sys`, or `js-sys` dependencies exist
   - No WASM build targets configured
   - The entire architecture assumes native execution

4. **Architecture Compatibility**
   - Using gloo would require:
     - Recompiling to WASM target
     - Running the entire LLM agent inside a browser
     - Restructuring vector stores, document loaders, etc.
     - This would be a complete rewrite, not a simple dependency swap

## Conclusion

The `headless_chrome` implementation is the **correct and only viable choice** for the `browser_use` tool in this context. Using `gloo` would be architecturally incompatible with the server-side nature of `langchain-ai-rust`.

If there's ever a need to port `langchain-ai-rust` to WASM for browser-based LLM applications, that would be a separate project with different design goals. In such a scenario, `gloo` would be appropriate for browser APIs, but you still wouldn't use it for "browser automation" in the same sense.

## References

- [gloo Documentation](https://gloo-rs.web.app/)
- [headless_chrome GitHub](https://github.com/rust-headless-chrome/rust-headless-chrome)
- [WebAssembly in Rust](https://rustwasm.github.io/)

## Status

**Resolved**: Documentation added to `src/tools/browser_use/browser_use.rs` explaining the architectural decision.

Date: 2026-02-08
