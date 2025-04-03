# My head canon for MCP resource vs tool

When learning about Model Context Protocol (MCP) I was confused about the reason for discriminating between a resource and a tool, isn't a tool strictly more powerful?

What clarified it for me was putting it in terms of HTTP:

> A resource is a GET request (stateless, URI only) while tool is a POST (possibly stateful, json payload)

Surprisingly I haven't seen this analogy been made before, instead most discussion is around server-client control / responsibility. There might be more nuance here that I'm missing.

---

Similarly, for HTTP you could make the reductionist argument about http that we actually don’t need GET requests at all, since a POST is just a GET with payload. Yet we use GET everyday for most tasks.
