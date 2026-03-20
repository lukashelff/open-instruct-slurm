
def extract_code_block(text: str) -> str:
    """
    If the text contains [RULE]...[/RULE] (or variants) or fenced code blocks,
    return the last block's content. Otherwise return the original text.
    """
    if not isinstance(text, str):
        return text

    # Remove leading "thinking" content if present
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # [RULE] ... [/RULE] variants, including [\RULE], [ /RULE]
    rule_blocks = re.findall(r"\[RULE\]\s*(.*?)\s*\[\s*\\?/RULE\s*\]", text, re.DOTALL | re.IGNORECASE)
    if rule_blocks:
        return rule_blocks[-1].strip()

    # Fenced code blocks ```...```
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Fallback: split on common "final answer" markers
    markers = [
        "### Final Answer:",
        "Final Answer:",
        "### Final:",
        "Final:",
        "Answer:",
        "Final rule:",
        "Rule:",
    ]
    lower_text = text.lower()
    for marker in markers:
        idx = lower_text.rfind(marker.lower())
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text


def parse_complex(text):
    '''
    Extracts all facts and rules from the text.
    Unlike v2, this extracts ALL Prolog syntax regardless of predicate.
    This allows shortcuts like eastbound(train1). to be accepted by local judge.
    
    Args:
        text (str): The text to extract the ILP from.
    Returns:
        str: The ILP containing all facts and rules found.
    Examples:
        >>> parse_rule_v3("eastbound(train0).")
        "eastbound(train0)."
        >>> parse_rule_v3("eastbound(T) :- has_car(T, C). eastbound(train1).")
        "eastbound(T) :- has_car(T, C).\neastbound(train1)."
    '''
    # Prefer fenced code blocks if present
    has_code_block = bool(re.search(r"```", text))
    text = extract_code_block(text)
    text = re.sub(r'%.*?(?=\n|$)', '', text) # remove comments
    # If no code block is present, try a strict line-based extraction first
    # to avoid pulling facts from natural language sentences.
    if not has_code_block:
        strict_rule_pattern = r'(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)\s*$'
        strict_fact_pattern = r'(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)\s*$'
        strict_rules = re.findall(strict_rule_pattern, text)
        strict_facts = re.findall(strict_fact_pattern, text)
        if strict_rules or strict_facts:
            p_code = ''
            for rule in strict_rules:
                statement = rule.strip()
                if not statement.endswith('.'):
                    statement += '.'
                p_code += statement + '\n'

            for fact in strict_facts:
                statement = fact.strip()
                if not statement.endswith('.'):
                    statement += '.'
                # Exclude facts that appear inside any rule (head or body)
                is_part_of_rule = False
                fact_without_dot = statement.rstrip('.')
                for rule in strict_rules:
                    fact_normalized = fact_without_dot.replace(' ', '')
                    rule_normalized = rule.replace(' ', '')
                    if fact_normalized in rule_normalized:
                        is_part_of_rule = True
                        break
                if not is_part_of_rule:
                    p_code += statement + '\n'

            return p_code.strip()

    # Pre-process: collapse code blocks to single lines
    text = re.sub(r'\n\s*', ' ', text)  # crude: flatten all to one line
    
    p_code = ''
    
    # Pattern 1: Extract rules (with :- body)
    # Matches: predicate(args) :- body.
    rule_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)'
    rules = re.findall(rule_pattern, text)
    for rule in rules:
        statement = rule.strip()
        if not statement.endswith('.'):
            statement += '.'
        p_code += statement + '\n'
    
    # Pattern 2: Extract facts (no :- body)
    # Matches: predicate(args).
    # But exclude facts that are part of rules (already captured above)
    fact_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)'
    facts = re.findall(fact_pattern, text)
    rule_statements_set = set(rule.strip() for rule in rules)
    
    for fact in facts:
        statement = fact.strip()
        if not statement.endswith('.'):
            statement += '.'
        # Only add if it's a pure fact (not part of any rule we already captured)
        # A fact is part of a rule if it appears anywhere in that rule (head or body)
        is_part_of_rule = False
        fact_without_dot = statement.rstrip('.')
        
        for rule in rules:
            # Check if this fact appears anywhere in the rule
            # Remove spaces for comparison to handle whitespace differences
            fact_normalized = fact_without_dot.replace(' ', '')
            rule_normalized = rule.replace(' ', '')
            
            # Check if the fact (without the final dot) appears in the rule
            # This catches both head and body occurrences
            if fact_normalized in rule_normalized:
                is_part_of_rule = True
                break
        
        # Only add standalone facts (not part of any rule)
        if not is_part_of_rule:
            p_code += statement + '\n'
    
    return p_code.strip()



def parse_simple(text):
    """
    Extracts all facts and rules from the text.
    Unlike v2, this extracts ALL Prolog syntax regardless of predicate.
    This allows shortcuts like eastbound(train1). to be accepted by local judge.

    Args:
        text (str): The text to extract the ILP from.
    Returns:
        str: The ILP containing all facts and rules found.
    Examples:
        >>> extract_ilp_from_text("eastbound(train0).")
        "eastbound(train0)."
        >>> extract_ilp_from_text("eastbound(T) :- has_car(T, C). eastbound(train1).")
        "eastbound(T) :- has_car(T, C).\neastbound(train1)."
    """
    text = re.sub(r"%.*?(?=\n|$)", "", text)  # remove comments
    # Pre-process: collapse code blocks to single lines
    text = re.sub(r"\n\s*", " ", text)  # crude: flatten all to one line

    p_code = ""

    # Pattern 1: Extract rules (with :- body)
    # Matches: predicate(args) :- body.
    rule_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)"
    rules = re.findall(rule_pattern, text)
    for rule in rules:
        statement = rule.strip()
        if not statement.endswith("."):
            statement += "."
        p_code += statement + "\n"

    # Pattern 2: Extract facts (no :- body)
    # Matches: predicate(args).
    # But exclude facts that are part of rules (already captured above)
    fact_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)"
    facts = re.findall(fact_pattern, text)
    rule_statements_set = set(rule.strip() for rule in rules)

    for fact in facts:
        statement = fact.strip()
        if not statement.endswith("."):
            statement += "."
        # Only add if it's a pure fact (not part of any rule we already captured)
        # A fact is part of a rule if it appears anywhere in that rule (head or body)
        is_part_of_rule = False
        fact_without_dot = statement.rstrip(".")

        for rule in rules:
            # Check if this fact appears anywhere in the rule
            # Remove spaces for comparison to handle whitespace differences
            fact_normalized = fact_without_dot.replace(" ", "")
            rule_normalized = rule.replace(" ", "")

            # Check if the fact (without the final dot) appears in the rule
            # This catches both head and body occurrences
            if fact_normalized in rule_normalized:
                is_part_of_rule = True
                break

        # Only add standalone facts (not part of any rule)
        if not is_part_of_rule:
            p_code += statement + "\n"

    return p_code.strip()