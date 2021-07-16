import React, { useRef, useState } from "react";
import ReactDOM from "react-dom";
import Editor from "@monaco-editor/react";
import Select from "react-select";
import _ from "lodash";

import * as wasm from "psiop-wasm";
import "../index.css";

const EXAMPLES = [
  {
    label: "Cancer",
    program: `has_cancer := flip(1/1000); 
if has_cancer {
  p_test_positive := 9/10
} else {
  p_test_positive := 1/10
};
test_positive := flip(p_test_positive)`,
  },
];

let App = () => {
  let editor = useRef(null);

  let [output, set_output] = useState("");
  let on_change = () => {
    let program = editor.current.getValue();
    try {
      set_output(wasm.get_dist(program));
    } catch (e) {
      set_output(e.toString());
    }
  };

  let ExampleSelector = () => (
    <div className="mb-8">
      Examples:{" "}
      <div className="w-60 inline-block">
        <Select
          options={EXAMPLES.map(({ label }, i) => ({ label, value: i }))}
          onChange={({ label }) => {
            editor.current.setValue(_.find(EXAMPLES, { label }).program);
          }}
        />
      </div>
    </div>
  );

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl mb-4">Psiop demo</h1>
      <ExampleSelector />
      <div className="grid grid-cols-2">
        <div>
          <Editor
            height="90vh"
            defaultValue={EXAMPLES[0].program}
            onMount={(ed) => {
              editor.current = ed;
              on_change();
            }}
            onChange={on_change}
            options={{
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
            }}
          />
        </div>
        <pre>{output}</pre>
      </div>
    </div>
  );
};

ReactDOM.render(<App />, document.getElementById("app"));
