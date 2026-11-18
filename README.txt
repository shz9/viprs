VIPRS browser application

Serve this directory over HTTP, e.g. python -m http.server 8000, then open
http://localhost:8000/webapp/. Do not open index.html via file://.
Pyodide runtime packages and the example data require internet access.
All four required wheels are included under webapp/wheels/.
The runtime manifest pins their SHA-256 hashes; build-info.json records
package versions, source commit/hashes, and compiler information.
To install in the full documentation site, copy webapp/ into docs/.
Documentation: https://shz9.github.io/viprs/web_application/
