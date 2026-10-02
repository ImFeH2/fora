ACCESS_ERROR_PAGE = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Access to Fora requires authentication</title>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  background: rgb(31 30 30); color: rgb(247 245 244);
  font: 14px/20px Inter, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
  letter-spacing: 0;
}
main { display: flex; min-height: 100%; padding: 24px; overflow-wrap: anywhere; }
.access-notice {
  display: grid; grid-template-columns: 16px minmax(0, 1fr);
  gap: 2px 8px; width: 100%; max-width: 520px; margin: auto;
  padding: 12px 14px; border: 1px solid rgb(228 108 99 / 32%);
  border-radius: 8px; background: rgb(228 108 99 / 4%);
}
.notice-mark {
  grid-row: span 2; display: grid; width: 16px; height: 16px; place-items: center;
  margin-top: 2px; border: 1.5px solid rgb(228 108 99); border-radius: 50%;
  color: rgb(228 108 99); font-size: 12px; font-weight: 600; line-height: 1;
}
h1 { margin: 0; font: inherit; font-weight: 500; }
.description { grid-column: 2; display: flex; min-width: 0; flex-direction: column; gap: 10px; }
p { margin: 0; color: rgb(175 172 171); }
</style>
</head>
<body>
<main>
<section class="access-notice" role="alert" aria-labelledby="title" aria-describedby="description">
<span class="notice-mark" aria-hidden="true">!</span>
<h1 id="title">Access to Fora requires authentication</h1>
<div class="description" id="description">
<p>The access credentials are missing or no longer valid.</p>
<p>Open Fora from the application, or use the access link provided when Fora starts.
That link includes the credentials needed to connect.</p>
</div>
</section>
</main>
</body>
</html>
"""
