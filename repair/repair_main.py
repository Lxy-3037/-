import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                          T5ForConditionalGeneration, RobertaTokenizer)
from tqdm.auto import tqdm
import json
import tree_sitter
from tree_sitter import Language



def remove_comments_and_strings(source_code):
    # 移除所有多行注释
    no_multiline_comments = re.sub(re.compile(r'/\*.*?\*/', re.DOTALL), '', source_code)
    # 移除所有单行注释
    no_single_comments = re.sub(re.compile(r'//.*?\n'), '\n', no_multiline_comments)
    # 移除所有字符串
    no_strings = re.sub(re.compile(r'"(\\.|[^"\\])*"'), '', no_single_comments)
    return no_strings

def find_pointer_declarations(source_code):
    # 首先，移除代码中的注释和字符串
    code = remove_comments_and_strings(source_code)
    
    # 用于匹配指针声明语句的正则表达式模式
    # basic_pointer_declaration_pattern = r'\b(\w+)\s+((?:\*+\s*)+)\s*(\w+)\s*(?:=\s*[^;]+)?\s*;'
    basic_pointer_declaration_pattern = r'\b(\w+)\s*(\*+)\s*(\w+)\s*(?:=\s*[^;]*\s*)?;'
    func_pointer_declaration_pattern = r'\b\w+\s*\(\*\s*(\w+)\s*\)\s*(\([^()]*\))\s*;'
    func_2_pointer_declaration_pattern = r'\w+\s*\*\s*\(\*\s*(\w+)\s*\)\s*\(\s*[^()]*\s*,\s*[^()]*\s*\);'
    func_parameter_pointer_pattern = r'\b(\w+)\s+((?:\*+\s*)+)\s*(\w+)\s*(?:\[[^\]]*\])?\s*(?:,|\))'



    # 在处理过的代码中查找指针声明语句
    basic_pointer_declarations = re.findall(basic_pointer_declaration_pattern, code)
    func_pointer_declarations = re.findall(func_pointer_declaration_pattern, code)
    func_2_pointer_declarations = re.findall(func_2_pointer_declaration_pattern, code)
    func_parameter_pointers = re.findall(func_parameter_pointer_pattern, code)

    def parse_parameters(matches):
        parsed_matches = []
        for match in matches:
            if len(match) == 3:
                param_type = match[0]
                pointer = re.sub(r'\s+', '', match[1])  # 去除额外的空格
                param_name = match[2]
                parsed_matches.append((param_type, pointer, param_name))
        return parsed_matches
    
    func_parameter_pointers = parse_parameters(func_parameter_pointers)

     # 创建一个集合来存储所有的 var_name
    all_var_names = set()

    # 将找到的 var_name 添加到集合中
    for (_, stars, var_name) in basic_pointer_declarations:
        all_var_names.add(var_name)
    for (var_name, _) in func_pointer_declarations:
        all_var_names.add(var_name)
    for var_name in func_2_pointer_declarations:
        all_var_names.add(var_name)
    for (_, stars, var_name) in func_parameter_pointers:
        all_var_names.add(var_name)
    # 输出找到的指针声明
    # print("func_parameter_pointers:", func_parameter_pointers)
    # print("basic_pointer_declarations:", basic_pointer_declarations)
    # print("func_pointer_declarations:", func_pointer_declarations)
    # print("func_2_pointer_declarations:", func_2_pointer_declarations)

    # 输出找到的指针声明
    # print("basic_pointer_declarations:", [var_name if stars == '*' else '*' + var_name for (_, stars, var_name) in basic_pointer_declarations])
    # print("func_pointer_declarations:", [var_name for (var_name, _) in func_pointer_declarations])
    # print("func_2_pointer_declarations:", ['*' + var_name for var_name in func_2_pointer_declarations])
    # print("func_parameter_pointers:", [var_name if stars == '*' else '*' + var_name for (_, stars, var_name) in func_parameter_pointers])

    # 输出找到的 var_name
    # print("All var_names:", all_var_names)
    return all_var_names

def find_pointer_declarations_2(source_code):
    code = remove_comments_and_strings(source_code)
    # * 匹配的指针
    star_del_pat = r'\*\s*(\w+)'
    # -> 匹配的指针
    der_del_pat = r'(\b\w+)\s*->'
    # 数组索引
    arr_del_pat = r"(\w+)\s+(\w+)\s*\[\s*\d*\s*\]\s*;"
    # 特殊指针
    special_del_pat = r'\(\s*\(?[^)]*\*\s*\)\s*([a-zA-Z_][a-zA-Z0-9_]*)'

    star_del = re.findall(star_del_pat, code)
    der_del = re.findall(der_del_pat, code)
    arr_del = re.findall(arr_del_pat, code)
    special_del = re.findall(special_del_pat, code)

    # print("star_del",star_del)
    # print("der_del",der_del)
    # print("arr_del",arr_del)

    all_vars = set()
    for(var_name) in star_del: all_vars.add(var_name)
    for(var_name) in der_del: all_vars.add(var_name)
    for (_ , var_name) in arr_del: all_vars.add(var_name)
    for(var_name) in special_del: all_vars.add(var_name)

    return all_vars

def extract_and_filter_variable_names(code):
    # 定义C/C++的保留字和专有名词列表
    c_cpp_keywords = [
        'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern',
        'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
        'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while', 'asm', 'bool', 'catch',
        'class', 'const_cast', 'delete', 'dynamic_cast', 'explicit', 'export', 'false', 'friend', 'inline', 'mutable',
        'namespace', 'new', 'operator', 'private', 'protected', 'public', 'reinterpret_cast', 'static_cast', 'template',
        'this', 'throw', 'true', 'try', 'typeid', 'typename', 'using', 'virtual', 'wchar_t', 'main', 'NULL', 'nullptr'
    ]
    
    def extract_variable_names(cleaned_code):
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'  # 匹配变量名的正则表达式
        return re.findall(pattern, cleaned_code)

    # 首先移除代码中的注释和字符串
    cleaned_code = remove_comments_and_strings(code)
    # 提取变量名
    variable_names = extract_variable_names(cleaned_code)
    # 过滤掉保留字和专有名词
    filtered_variable_names = [name for name in variable_names if name not in c_cpp_keywords]
    
    return set(filtered_variable_names)

def find_first_occurrence(source_code, variable_name):
    # 将代码中的注释去除，以避免注释影响匹配
    code_without_comments = re.sub(r'//.*|/\*.*?\*/', '', source_code, flags=re.DOTALL)
    # 将代码中的字符串文字去除，以避免字符串文字中的变量名被误认为是变量
    code_without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '', code_without_comments)
    # 找到变量名第一次出现的位置
    match = re.search(r'\b%s\b' % re.escape(variable_name), code_without_strings)
    if match:
        # 获取变量名的起始位置和结束位置
        start_index = match.start()
        end_index = match.end()
        # 无需检查变量名是否被空格包围，直接扩展以获取整个语句
        left_index = start_index
        while left_index > 0 and code_without_strings[left_index-1] not in ";{}":
            left_index -= 1
        right_index = end_index
        while right_index < len(code_without_strings) and code_without_strings[right_index] not in ";{}":
            right_index += 1
        # 包括语句结束的分号
        if right_index < len(code_without_strings) and code_without_strings[right_index] == ';':
            right_index += 1
        # line_number = code_without_strings.count('\n', 0, left_index) + 1
        line_content = code_without_strings[left_index:right_index].strip()
        return line_content

    return None


def find_pattern_matches_in_code(code, patterns):
    """
    Find all unique matches for given patterns in the provided code segment.
    """
    matches_set = set()
    for pattern in patterns:
        matches = re.findall(pattern, code)
        matches_set.update(matches)
    return matches_set


def locate_vulnerabilities_in_source_code(source_code):
    """
    Locate the vulnerability code segments surrounded by <S2SV_StartBug> and <S2SV_EndBug> in the given source code.
    """
    start_tag = "<S2SV_StartBug>"
    end_tag = "<S2SV_EndBug>"
    vulnerability_codes = []

    start_index = source_code.find(start_tag)
    while start_index != -1:
        end_index = source_code.find(end_tag, start_index)
        if end_index != -1:
            vulnerability_code = source_code[start_index + len(start_tag): end_index].strip()
            vulnerability_codes.append(vulnerability_code)
            start_index = source_code.find(start_tag, end_index)
        else:
            break

    return vulnerability_codes

def locate_fixes_in_target_code(source_code):
    """
    Locate the vulnerability code segments surrounded by <S2SV_StartBug> and <S2SV_EndBug> in the given source code.
    """
    start_tag = "<S2SV_ModStart>"
    end_tag = "<S2SV_ModEnd>"
    fix_codes = []

    start_index = source_code.find(start_tag)
    while start_index != -1:
        end_index = source_code.find(end_tag, start_index)
        if end_index != -1:
            fix_code = source_code[start_index + len(start_tag): end_index].strip()
            fix_codes.append(fix_code)
            start_index = source_code.find(start_tag, end_index)
        else:
            break

    return fix_codes

def identify_potential_null_dereferences(code):
    """
    Identify potential null pointer dereferences in the provided code segment.
    """
    # Patterns to find variable names used in dereferences
    dereference_patterns = [
        r'(\b\w+)\s*->',  # Matches "pointer->" and captures "pointer"
    ]

    # Find all variables that are dereferenced
    dereferenced_vars = find_pattern_matches_in_code(code, dereference_patterns)
    return dereferenced_vars

def add_null_pointer_condition(vulnerability_code, dereferenced_vars):
    """
    Add null pointer check conditions to the provided code segment for the identified variables.
    """
    null_checks = []
    for var in dereferenced_vars:
        null_checks.append(f"if ({var} == NULL) return;")
    
    # Combine null checks with the original vulnerability code
    modified_code = '\n'.join(null_checks + [vulnerability_code])
    return modified_code

def generate_suggested_fix(source_code, model, tokenizer, device='cpu'):
    """
    Generate suggested fixes for a vulnerability in the source code using a pretrained model.
    """
    # Prepare the source code for the model
    encoded_input = tokenizer.encode_plus(source_code, return_tensors="pt", max_length=512, truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate the output
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512)

    # Decode the generated ids to get the text
    suggested_fix = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return suggested_fix

def generate_syntax_tree(source_code):
    # 加载 C 语言的语法
    C_LANGUAGE = Language('/home/xiyang/tools/tree_sitter/build/my-languages.so', 'c')
    # 创建 C 语言的语法分析器
    parser = tree_sitter.Parser()
    parser.set_language(C_LANGUAGE)
    # 解析源代码并生成语法树
    tree = parser.parse(bytes(source_code, "utf8"))
    # 构建语法树的JSON表示形式
    syntax_tree = construct_json_tree(tree.root_node, source_code)
    return syntax_tree

def construct_json_tree(node, source_code):
    if node.child_count == 0:
        node_content = source_code[node.start_byte:node.end_byte]
        node_type = node.type
        if node_type == "pointer_declarator":
            # 对指针变量节点进行特殊处理，记录指针的类型和标识符
            pointer_type = None
            identifier = None
            for child in node.children:
                if child.type == "type_identifier":
                    pointer_type = child.content
                elif child.type == "identifier":
                    identifier = child.content
            return {"type": node_type, "pointer_type": pointer_type, "identifier": identifier, "content": node_content}
        else:
            return {"type": node_type, "content": node_content}
    else:
        children = [construct_json_tree(child, source_code) for child in node.children]
        return {"type": node.type, "children": children}


def get_variable_type(node, variable_name, source_code):
    """
    在语法树中递归搜索变量的声明以找到其类型。
    """
    if 'children' not in node:
        return None

    for child in node['children']:
        if child['type'] == "parameter_declaration" or child['type'] == "declaration":
            # 在声明节点中寻找类型和标识符
            identifier = None
            type_identifier = None
            for c in child['children']:
                if c['type'] == "identifier" and c.get("content") == variable_name:
                    identifier = c
                elif c['type'] == "type_identifier":
                    type_identifier = c
                elif c['type'] == "pointer_declarator":
                    # 对于指针声明，我们需要特别处理以获取正确的类型
                    pointer_type = "pointer"
                    # 检查是否有类型标识符
                    if 'children' in c:
                        for pointer_child in c['children']:
                            if pointer_child['type'] == "type_identifier":
                                type_identifier = pointer_child
                                break
                    if type_identifier:
                        return f"{type_identifier.get('content')} {pointer_type}"

            if identifier and type_identifier:
                return type_identifier.get("content")

        found_type = get_variable_type(child, variable_name, source_code)
        if found_type:
            return found_type

    return None


def analyze_vulnerability_code_syntax(source_code, vulnerability_code, syntax_tree):
    """
    Analyze the syntax of the vulnerability code to check for potential null pointer dereferences.
    """
    # Patterns to find variable names used in dereferences
    dereference_patterns = [
        r'(\b\w+)\s*->',  # Matches "pointer->" and captures "pointer"
        r'\*\s*(\w+)',    # Matches "*pointer" and captures "pointer"
    ]

    # Find all variables that are dereferenced
    dereferenced_vars = find_pattern_matches_in_code(vulnerability_code, dereference_patterns)

    # Analyze variable types using syntax tree
    variable_types = {}
    for var in dereferenced_vars:
        variable_type = get_variable_type(syntax_tree, var, source_code)
        if variable_type:
            variable_types[var] = variable_type

    return variable_types


def check_parentheses(divisor):
    """
    Check if parentheses are balanced and remove unbalanced ones.
    """
    count = 0
    balanced_divisor = ''
    
    for char in divisor:
        if char == '(':
            count += 1
            balanced_divisor += char
        elif char == ')':
            if count > 0:
                count -= 1
                balanced_divisor += char
        else:
            balanced_divisor += char
    
    return balanced_divisor

def recursively_process_divisors(divisors):
    """
    Recursively process divisors to handle nested division operations while keeping all levels of divisors.
    """
    processed = set()  # Use a set to avoid duplicates
    result = []  # List to store the final divisors

    def process_divisor(divisor):
        # This function processes each divisor recursively and adds it to the result
        # Check for further division operations within the divisor
        sub_divisors = find_pattern_matches_in_code(divisor, [r'[^=/*+-]*?/\s*([^;\n]+)'])
        
        if sub_divisors:
            # If there are sub-divisors, process each of them recursively
            for sub in sub_divisors:
                process_divisor(sub)
        # Add the original divisor to the result after checking parentheses
        clean_divisor = check_parentheses(divisor)
        if clean_divisor and clean_divisor not in processed:
            processed.add(clean_divisor)
            result.append(clean_divisor)

    for divisor in divisors:
        process_divisor(divisor)

    return result

def find_divisors(code):
    """
    Uses find_pattern_matches_in_code to find all divisors in division operations within the given code.
    """
    patterns = [r'[^=/*+-]*?/\s*([^;\n]+)']

    code = remove_comments_and_strings(code)

    matches = find_pattern_matches_in_code(code, patterns)

    # Preprocess all matches to remove any with unbalanced parentheses
    preprocessed_matches = [check_parentheses(match.strip()) for match in matches]
    # Recursively process these divisors
    divisors = recursively_process_divisors(preprocessed_matches)
    
    return divisors

def generate_division_safety_check(divisors, vulnerability_code):
    zero_checks = []
    for var in divisors:
        zero_checks.append(f"if ({var} == 0) {{fprintf(stderr, \"Error: Division by zero in operation\\n\"); return -1;}}")
    
    # Combine null checks with the original vulnerability code
    modified_code = '\n'.join(zero_checks + [vulnerability_code])
    return modified_code
 
def find_assertions(code):
    """
    Find all assertion statements in the provided code segment.

    Args:
    code (str): The source code to analyze.

    Returns:
    list: A list of assertion statements.
    """
    # 定义查找断言的正则表达式模式
    # assertion_pattern = r'\bassert\s*\(([^)]+)\);'
    assertion_pattern = r'\b(assert\s*\(.*?\))'
    code = remove_comments_and_strings(code)
    # 调用 find_pattern_matches_in_code 函数查找断言
    assertions = find_pattern_matches_in_code(code, [assertion_pattern])

    return assertions


def convert_asserts_to_ifs(assertions):
    """
    Convert a list of assertion statements to a list of if statements with the opposite conditions.

    Args:
    assertions (list): A list of assertion statements to convert.

    Returns:
    list: A list of converted if statements.
    """
    if_statements = []
    
    for assertion in assertions:
        # 去除 "assert" 关键字、分号和括号
        condition = assertion.replace("assert", "").strip(" ;()")
        # 构建 if 语句
        if_statement = f"if (!({condition})) {{\n    return;\n}} \n// {assertion}"
        if_statements.append(if_statement)

    return if_statements

def convert_assert_to_if_v2(assertion):
    """
    Convert an assertion statement to an if statement with the opposite condition.

    Args:
    assertion (str): The assertion statement to convert.

    Returns:
    str: The converted if statement.
    """
    # 去除 "assert" 关键字、分号和括号
    condition = assertion.replace("assert", "").strip(" ;()")
    # 构建 if 语句
    if_statement = f"if (!({condition})) {{\n    return;\n}} \n//{assertion}"

    return if_statement


def repair_476(source_code, link, index, model, tokenizer, device='cpu'):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    fixed_code_list = []

    if vulnerability_codes:
        print(f"476 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
            # Identify potential null pointer dereferences
            dereferenced_vars = identify_potential_null_dereferences(vulnerability_code)
            # Extract and filter variable names
            variable_names = extract_and_filter_variable_names(vulnerability_code)
            # Remove dereferenced variables from variable names
            variable_names = variable_names - dereferenced_vars

            # Find first occurrence of variables in source code and check if they are pointers
            for variable_name in variable_names:
                # Find the first occurrence of the variable in the source code
                first_occurrence = find_first_occurrence(source_code, variable_name)
                if first_occurrence:
                    # Check if the statement contains pointer declarations for the variable
                    pointer_vars = find_pointer_declarations_2(first_occurrence)
                    print("variable_name:", variable_name)
                    print("first_occurrence:", first_occurrence)
                    print("pointer_declarations", pointer_vars)
                    print()
                    if variable_name in pointer_vars:
                        # Add pointer variable to set of dereferenced vars
                        dereferenced_vars.add(variable_name)

            # Check if any identified variables are potential null pointer dereferences
            if dereferenced_vars:
                # Add null pointer check for the variables
                fixed_code = add_null_pointer_condition(vulnerability_code, list(dereferenced_vars))
            else:
                fixed_code = generate_suggested_fix(source_code, model, tokenizer, device)
            
            fixed_code_list.append(fixed_code)
            print(f"Fixed code {i}:\n{fixed_code}\n")
            print("-" * 60)
    
    return fixed_code_list


def repair_476_v2(source_code, link, index):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    has_fixed = False  # 标志变量，用于跟踪是否生成了有效的补丁

    if vulnerability_codes:
        print(f"476 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
            # Identify potential null pointer dereferences
            dereferenced_vars = identify_potential_null_dereferences(vulnerability_code)
            # Extract and filter variable names
            variable_names = extract_and_filter_variable_names(vulnerability_code)
            # Remove dereferenced variables from variable names
            variable_names = variable_names - dereferenced_vars

            # Find first occurrence of variables in source code and check if they are pointers
            for variable_name in variable_names:
                # Find the first occurrence of the variable in the source code
                first_occurrence = find_first_occurrence(source_code, variable_name)
                if first_occurrence:
                    # Check if the statement contains pointer declarations for the variable
                    pointer_vars = find_pointer_declarations_2(first_occurrence)
                    print("variable_name:", variable_name)
                    print("first_occurrence:", first_occurrence)
                    print("pointer_declarations", pointer_vars)
                    print()
                    if variable_name in pointer_vars:
                        # Add pointer variable to set of dereferenced vars
                        dereferenced_vars.add(variable_name)

            # Generate a fixed code if there are any dereferenced variables needing null checks
            if dereferenced_vars:
                fixed_code = add_null_pointer_condition(vulnerability_code, list(dereferenced_vars))
                has_fixed = True  # 有效的补丁已生成
            else:
                fixed_code = ""  # 没有发现需要特别处理的空指针解引用，不生成修复

            if fixed_code:
                print(f"Fixed code {i}:\n{fixed_code}\n")
                print("-" * 60)
            else:
                print("No fix needed for this vulnerability.")
                print("-" * 60)
    
    return has_fixed


def repair_369(source_code, link, index, model, tokenizer, device='cpu'):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    fixed_code_list = []

    if vulnerability_codes:
        print(f"369 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
            # Identify potential zero divisors
            divisors = find_divisors(vulnerability_code)
            for divisor in divisors:
                print("divisor:", divisor)
            # Check if any identified variables are potential null pointer dereferences
            if divisors:
                # Add zero divisor check for the variables
                fixed_code = generate_division_safety_check(divisors, vulnerability_code)
            else:
                fixed_code = generate_suggested_fix(source_code, model, tokenizer, device)
            
            fixed_code_list.append(fixed_code)
            print(f"Fixed code {i}:\n{fixed_code}\n")
            print("-" * 60)
    
    return fixed_code_list


def repair_369_v2(source_code, link, index):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    has_fixed = False  # To track whether a valid patch was generated

    if vulnerability_codes:
        print(f"369 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
            # Identify potential zero divisors
            divisors = find_divisors(vulnerability_code)
            for divisor in divisors:
                print("divisor:", divisor)
            
            # Generate a fix if there are any divisors
            if divisors:
                fixed_code = generate_division_safety_check(divisors, vulnerability_code)
                has_fixed = True  # Mark that a valid patch was generated
            else:
                fixed_code = ""  # No divisors needing fix, so no code generated
            
            if fixed_code:
                print(f"Fixed code {i}:\n{fixed_code}\n")
                print("-" * 60)
            else:
                print("No fix needed for this vulnerability.")
                print("-" * 60)
    
    return has_fixed

def repair_617(source_code, link, index, model, tokenizer, device='cpu'):
    """
    Repair unreachable assertions in the provided source code.

    Args:
    source_code (str): The source code to analyze and repair.

    Returns:
    str: The repaired source code.
    """
    # Find assertion statements in the source code
    assertions = find_assertions(source_code)
    fixed_code_list = []

    if assertions:
        print(f"617 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, assertion in enumerate(assertions, 1):
            print(f"assertion {i}:\n{assertion}\n")
            if_statment = convert_assert_to_if_v2(assertion)
            print(f"Fixed code{i}:\n{if_statment}\n")
            fixed_code_list.append(if_statment)
    else:
        fixed_code = generate_suggested_fix(source_code, model, tokenizer, device)
        fixed_code_list.append(fixed_code)
        print(f"Fixed code 1:\n{fixed_code}\n")

    print("-" * 60)

    return fixed_code_list


def repair_617_v2(source_code, link, index):
    """
    Repair unreachable assertions in the provided source code.

    Args:
    source_code (str): The source code to analyze and repair.

    Returns:
    bool: True if any fixes were applied, otherwise False.
    """
    # Find assertion statements in the source code
    assertions = find_assertions(source_code)
    has_fixed = False  # To track whether any fixes were applied

    if assertions:
        print(f"617 Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}:\n")
        for i, assertion in enumerate(assertions, 1):
            print(f"assertion {i}:\n{assertion}\n")
            if_statement = convert_assert_to_if_v2(assertion)
            if if_statement:
                has_fixed = True  # A fix was generated
                print(f"Fixed code {i}:\n{if_statement}\n")
            else:
                print(f"No fix needed for assertion {i}.\n")
    else:
        print("No assertions found needing repair.\n")

    print("-" * 60)
    return has_fixed


def repair_125(source_code, target_code, link, index, model, tokenizer, device='cpu'):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    fixs_codes = locate_fixes_in_target_code(target_code)
    fixed_code_list = []
    print(f"source_code:{source_code}")

    if vulnerability_codes:
        print(f"Vulnerabilities found in row {index + 1}:\n")
        print(f"link: {link}\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
        for j, fix_code in enumerate(fixs_codes, 1):
            print(f"fix_code {j}:\n{fix_code}\n")
            
    return fixed_code_list

def repair(source_code, index, model, tokenizer, device='cpu'):
    vulnerability_codes = locate_vulnerabilities_in_source_code(source_code)
    fixed_code_list = []

    if vulnerability_codes:
        print(f"Vulnerabilities found in row {index + 1}:\n")
        for i, vulnerability_code in enumerate(vulnerability_codes, 1):
            print(f"Vulnerability {i}:\n{vulnerability_code}\n")
            
    fixed_code = generate_suggested_fix(source_code, model, tokenizer, device)  
    fixed_code_list.append(fixed_code)
    print(f"Fixed code:\n{fixed_code}\n")
    print("-" * 60)
    
    return fixed_code_list


def main():
    csv_file_path1 = '/home/xiyang/Documents/VulRepair/data/fine_tune_data/whole.csv'
    df1 = pd.read_csv(csv_file_path1)
  
    count_476 = 0
    fixed_476 = 0
    count_369 = 0
    fixed_369 = 0
    count_617 = 0
    fixed_617 = 0

    for index, row in df1.iterrows():
        source_code = row['source']
        link = row['original_address']

        if source_code.startswith("CWE-476"):
            count_476 += 1
            fixed = repair_476_v2(source_code, link, index)
            if fixed:
                fixed_476 += 1
        elif source_code.startswith("CWE-369"):
            count_369 += 1
            fixed = repair_369_v2(source_code, link, index)
            if fixed:
                fixed_369 += 1
        elif source_code.startswith("CWE-617"):
            count_617 += 1
            fixed = repair_617_v2(source_code, link, index)
            if fixed:
                fixed_617 += 1
    # Print statistics
    print(f"CWE-476: {count_476} found, {fixed_476} fixed.")
    print(f"CWE-369: {count_369} found, {fixed_369} fixed.")
    print(f"CWE-617: {count_617} found, {fixed_617} fixed.")

if __name__ == "__main__":
    main()



