import std/asynchttpserver
import std/asyncdispatch

proc main {.async.} =
    var server = newAsyncHttpServer()
    proc cb(req: Request) {.async.} =
        echo (req.reqMethod, req.url, req.headers)
        let headers = {"Content-type": "text/plain; charset=utf-8"}
        await req.respond(Http200, "Hello World", headers.newHttpHeaders())

    server.listen(Port(11434))
    let port = server.getPort
    echo "test this with: curl localhost:" & $port.uint16 & "/"
    while true:
        if server.shouldAcceptRequest():
            await server.acceptRequest(cb)
        else:
            # too many concurrent connections, `maxFDs` exceeded
            # wait 500ms for FDs to be closed
            await sleepAsync(500)

waitFor main()