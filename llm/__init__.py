import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def askgpt(user, system=None, model="gpt-4o", **kwargs):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return client.chat.completions.create(model=model, messages=msgs, **kwargs)

def response(compl):
    return compl.choices[0].message.content

def run(code):
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None
    
    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id='_result', ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename='<ast>', mode='exec'), ns)
    return ns.get('_result', None)

def python(code:str,debug=False):
    "Return result of executing `code` using python. If execution not permitted, returns `#FAIL#`"
    go = input(f'Proceed with execution?\n```\n{code}\n```\n')
    if go.lower()!='y': return '#FAIL#'
    return run(code)