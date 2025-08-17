const fs = require("fs");
const path = require("path");

const files = ["realtime.js", "control.js", "jobs.js"];
const dir = path.join(__dirname, "..", "lib", "proto");

for (const f of files) {
  const p = path.join(dir, f);
  if (!fs.existsSync(p)) continue;
  let txt = fs.readFileSync(p, "utf8");
  if (!/export default \$root;/.test(txt)) {
    txt += "\nexport default $root;\n";
    fs.writeFileSync(p, txt, "utf8");
  }
}
console.log("proto:fix done");