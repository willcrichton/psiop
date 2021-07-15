import esbuild from 'esbuild';
import {wasmLoader} from 'esbuild-plugin-wasm';
import fs from 'fs/promises';
import {program} from 'commander';

program.option('-w, --watch');
program.parse(process.argv);
const options = program.opts();

await esbuild.build({
  sourcemap: true,
  bundle: true,
  watch: options.watch,
  entryPoints: ['js/index.jsx'],
  outdir: 'build',
  plugins: [wasmLoader()],
  format: 'esm',
});

await fs.copyFile('index.html', 'build/index.html');
