"""Held stdlib controls of source-bound planner compaction, never model outputs.

The admitted worker supplies both complete production source byte strings. Only
three actual pure functions and their actual constants execute; the real client
class is compared by AST, not imported. Token predicates are controls, not an
actual tokenizer or useful-planning qualification. No default test collection.
"""
from __future__ import annotations
import ast
import json
import types
import unittest
from dataclasses import dataclass
from typing import Any, Callable

CURRENT = ORIGINAL = None
CURRENT_TREE = ORIGINAL_TREE = None
PURE_NAMES = {'_short', '_compact_context', 'compact_planner_prompt'}
CONSTANT_NAMES = {'MAX_PROMPT_BYTES', '_PLANNER_RULES', '_REPLY_ONLY_RULES',
                  '_STRUCTURED_REPLY_ONLY_RULES'}
MARKER = ('ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal current '
          'request and do not substitute an older task): ')

@dataclass(frozen=True)
class Message:
    role: str
    content: str

def configure(current_raw, original_raw):
    global CURRENT, ORIGINAL, CURRENT_TREE, ORIGINAL_TREE
    def load(raw):
        tree = ast.parse(raw)
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in PURE_NAMES:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(name, ast.Name) and name.id in CONSTANT_NAMES for name in node.targets):
                selected.append(node)
        module = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'json': json, 'Any': Any, 'Callable': Callable, 'ModelMessage': Message}
        exec(compile(module, '<admitted actual pure planner functions>', 'exec'), namespace)
        return types.SimpleNamespace(**namespace), tree
    CURRENT, CURRENT_TREE = load(current_raw)
    ORIGINAL, ORIGINAL_TREE = load(original_raw)

def messages(payload=None, *, directive=None, auxiliary=None):
    result = []
    if directive is not None:
        result.append(Message('user', MARKER + json.dumps(directive)))
    result.append(Message('user', json.dumps(payload or {'fresh_facts': {}, 'skills': []})))
    if auxiliary is not None:
        result.append(Message('user', json.dumps(auxiliary)))
    return tuple(result)

def context(prompt):
    return json.loads(prompt.split('\nContext:', 1)[1])

class PlannerBudgetControls(unittest.TestCase):
    def setUp(self):
        if CURRENT is None:
            raise RuntimeError('only the reviewed source-bound worker configures this fixture')

    def test_shorter_rules_fit_minimum_without_raising_limits(self):
        gate = lambda text: len(text.encode()) <= 620
        with self.assertRaisesRegex(ValueError, 'admitted request budget'):
            ORIGINAL.compact_planner_prompt(messages(), fits_prompt=gate)
        prompt = CURRENT.compact_planner_prompt(messages(), fits_prompt=gate)
        self.assertTrue(gate(prompt))
        self.assertEqual(CURRENT.MAX_PROMPT_BYTES, ORIGINAL.MAX_PROMPT_BYTES)
        self.assertEqual(context(prompt), {'skills': []})

    def test_literal_request_keeps_trailing_prohibition_and_unicode(self):
        text = 'Observe the Pokémon terrain carefully. ' * 10 + 'Do not mine or move.'
        directive = {'message_id': 'x'*120, 'kind': 'instruction', 'text': text,
                     'priority': 'high', 'status': 'delivered'}
        prompt = CURRENT.compact_planner_prompt(messages(directive=directive))
        self.assertEqual(context(prompt)['active_operator_directive'], directive)
        self.assertNotIn('operator_question', context(prompt))

    def test_irreducible_literal_request_refuses_without_truncation(self):
        text = 'Do not move. ' * 500
        data = messages(directive={'message_id': 'a', 'kind': 'instruction', 'text': text})
        ctx, _ = CURRENT._compact_context(data)
        self.assertEqual(ctx['active_operator_directive']['text'], text)
        with self.assertRaisesRegex(ValueError, 'admitted request budget'):
            CURRENT.compact_planner_prompt(data)

    def test_required_constraints_and_fallback_parameters_survive_pressure(self):
        required = {'forbidden_inputs': ['w', 'a', 's', 'd'], 'sneak': True,
                    'nested': {'camera': {'max': 12, 'direction': 'left'}}}
        fallback = {'s': 'survey', 'p': {'hold_ms': 120, 'camera_only': True}, 'x': False}
        aux = {'authority_bounds': {'allowed_skills': [{'s': 'survey', 'p': ['hold_ms']}],
                'requested_skill_ids': ['survey'], 'required_action_constraints': required},
               'safe_fallback': fallback}
        captured = []
        def fits(text): captured.append(context(text)); return len(captured) >= 2
        CURRENT.compact_planner_prompt(messages({'fresh_facts': {}, 'skills': [],
            'recent_skill_runs': [{'skill': 'old', 'outcome': 'failed'}]}, auxiliary=aux), fits_prompt=fits)
        for row in captured:
            self.assertEqual(row['authority_bounds']['required_action_constraints'], required)
            self.assertEqual(row['safe_fallback'], fallback)

    def test_required_fields_over_eight_are_lossless(self):
        required = {f'constraint_{i}': 'do-not-delete-'+str(i) for i in range(12)}
        fallback = {'s': 'survey', 'p': {f'parameter_{i}': i for i in range(12)}, 'x': False}
        data = messages(auxiliary={'authority_bounds': {'required_action_constraints': required},
                                   'safe_fallback': fallback})
        ctx, _ = CURRENT._compact_context(data)
        self.assertEqual(ctx['authority_bounds']['required_action_constraints'], required)
        self.assertEqual(ctx['safe_fallback'], fallback)

    def test_requested_skill_outside_initial_menu_is_preserved(self):
        wanted = 'required-'+'s'*85
        parameter = 'parameter-'+'p'*70
        skills = [{'skill_id': f'optional{i}', 'parameters': []} for i in range(8)]
        skills.append({'skill_id': wanted, 'parameters': [parameter]})
        data = messages({'fresh_facts': {}, 'skills': skills}, auxiliary={'authority_bounds': {
            'requested_skill_ids': [wanted], 'allowed_skills': [{'s': wanted, 'p': [parameter]}]}})
        ctx, _ = CURRENT._compact_context(data)
        self.assertEqual(ctx['skills'][0]['skill_id'], wanted)
        self.assertEqual(ctx['skills'][0]['parameters'], [parameter])
        self.assertEqual(ctx['authority_bounds']['allowed_skills'], [{'s': wanted, 'p': [parameter]}])

    def test_authority_goal_outside_initial_menu_keeps_exact_id(self):
        wanted = 'operator:'+'g'*120
        goals = [{'id': 'old1'}, {'id': 'old2'}, {'id': wanted, 'description': 'Current goal'}]
        data = messages({'fresh_facts': {}, 'skills': [], 'goals': goals}, auxiliary={
            'authority_bounds': {'authority_goal_id': wanted}})
        ctx, _ = CURRENT._compact_context(data)
        self.assertEqual(ctx['goals'][0]['id'], wanted)
        self.assertEqual(ctx['authority_bounds']['authority_goal_id'], wanted)

    def test_menu_reduction_never_removes_requested_ids_or_parameters(self):
        needed = {'s': 'survey', 'p': ['camera_only', 'max_steps']}
        aux = {'authority_bounds': {'requested_skill_ids': ['survey'],
                'allowed_skills': [needed, {'s': 'optional', 'p': ['unused']}]}}
        data = messages(auxiliary=aux); captured = []
        def fits(text):
            row = context(text); captured.append(row)
            return len(row['authority_bounds']['allowed_skills']) == 1
        prompt = CURRENT.compact_planner_prompt(data, fits_prompt=fits)
        self.assertEqual(context(prompt)['authority_bounds']['allowed_skills'], [needed])
        for row in captured:
            self.assertIn(needed, row['authority_bounds']['allowed_skills'])
            self.assertEqual(row['authority_bounds']['requested_skill_ids'], ['survey'])

    def test_structured_reply_rules_match_full_wire_grammar(self):
        fmt = {'mode': 'operator_reply', 'authority': {'authority_goal_id': 'operator:current'}}
        data = messages(directive={'message_id': 'current', 'kind': 'question', 'text': 'What can you see?'})
        prompt = CURRENT.compact_planner_prompt(data, response_format=fmt)
        self.assertIn('r,g,s,p,o,c,x,q,w,d,n', prompt)
        self.assertNotIn('keys g, o and q', prompt)
        self.assertEqual(context(prompt)['reply_only_goal_id'], 'operator:current')
        self.assertEqual(fmt, {'mode': 'operator_reply', 'authority': {'authority_goal_id': 'operator:current'}})

    def test_legacy_reply_keeps_three_key_contract(self):
        prompt = CURRENT.compact_planner_prompt(messages(directive={
            'message_id': 'q', 'kind': 'question', 'text': 'What is nearby?'}))
        # The accepted client predates q and uses g/o; current main uses g/o/q.
        # Compaction must preserve its own original client's exact grammar.
        self.assertEqual(CURRENT._REPLY_ONLY_RULES, ORIGINAL._REPLY_ONLY_RULES)
        self.assertTrue(prompt.startswith(ORIGINAL._REPLY_ONLY_RULES))
        self.assertEqual(context(prompt)['reply_only_goal_id'], 'operator:q')

    def test_explicit_contract_mode_cannot_be_changed_by_prose(self):
        data = (Message('system', 'this response grants no game action authority'),
                Message('user', json.dumps({'fresh_facts': {}, 'skills': []})))
        prompt = CURRENT.compact_planner_prompt(data, response_format={
            'mode': 'plan', 'authority': {}})
        self.assertIn('Minecraft planner.', prompt)
        self.assertNotIn('reply_only_goal_id', context(prompt))

    def test_original_capsule_authority_precedes_repair_and_menu_selection(self):
        skills = [{'skill_id': f'old{i}', 'parameters': []} for i in range(8)]
        skills.append({'skill_id': 'survey', 'parameters': ['camera_only']})
        fmt = {'mode': 'plan', 'authority': {'goal_ids': ['operator:now'],
            'authority_goal_id': 'operator:now', 'requested_skill_ids': ['survey'],
            'allowed_skills': [{'skill_id': 'survey', 'parameters': ['camera_only']}],
            'required_action_constraints': {'camera_only': True}}}
        before = json.dumps(fmt, sort_keys=True)
        data = messages({'fresh_facts': {}, 'skills': skills,
            'goals': [{'id': 'old1'}, {'id': 'old2'}, {'id': 'operator:now'}]},
            auxiliary={'authority_bounds': {'authority_goal_id': 'old1',
                'requested_skill_ids': ['old1'], 'required_action_constraints': {'camera_only': False}}})
        prompt = CURRENT.compact_planner_prompt(data, response_format=fmt)
        ctx = context(prompt)
        self.assertEqual(ctx['goals'][0]['id'], 'operator:now')
        self.assertEqual(ctx['skills'][0]['skill_id'], 'survey')
        self.assertEqual(ctx['authority_bounds']['required_action_constraints'], {'camera_only': True})
        self.assertEqual(json.dumps(fmt, sort_keys=True), before)

    def test_legacy_active_instruction_retains_its_existing_goal_under_pressure(self):
        payload = {'fresh_facts': {}, 'skills': [],
            'goals': [{'id': 'old1'}, {'id': 'old2'}, {'id': 'operator:now'}]}
        captured = []
        def fits(text):
            ctx = context(text); captured.append(ctx)
            return len(ctx['goals']) == 1
        prompt = CURRENT.compact_planner_prompt(messages(payload, directive={
            'message_id': 'now', 'kind': 'instruction', 'text': 'Observe; do not move.'}), fits_prompt=fits)
        self.assertEqual(context(prompt)['goals'][0]['id'], 'operator:now')
        for ctx in captured:
            self.assertIn('operator:now', [row['id'] for row in ctx['goals']])

    def test_safety_facts_remain_when_optional_facts_are_removed(self):
        safety = {'danger.immediate': [False, 1], 'danger.drowning': [False, 1],
                  'danger.burning': [False, 1], 'scene.death': [False, 1],
                  'scene.playable': [True, 1], 'player.critical_health': [False, 1],
                  'environment.underwater': [False, 1]}
        payload = {'fresh_facts': {**safety, 'terrain.optional': ['x'*80, .8]}, 'skills': []}
        def fits(text): return 'terrain.optional' not in context(text)['fresh_facts']
        prompt = CURRENT.compact_planner_prompt(messages(payload), fits_prompt=fits)
        self.assertEqual(context(prompt)['fresh_facts'], safety)

    def test_answer_retains_source_and_exact_current_question(self):
        question = 'Please tell me '+('exactly '*45)+'what the evidence establishes?'
        source = {'title': 'Poké Ball', 'extract': '4 Red Apricorn and1 Copper Ingot.',
                  'url': 'https://minecraft.wiki/w/Copper_Ingot', 'version': 'pack:1.3.152', 'confidence': 1.0}
        payload = {'fresh_facts': {}, 'skills': [], 'wiki_evidence': [source, {'title': 'old'}]}
        def fits(text): return len(context(text)['wiki_evidence']) == 1
        prompt = CURRENT.compact_planner_prompt(messages(payload,directive={
            'message_id': 'q', 'kind': 'question', 'text': question}),fits_prompt=fits)
        self.assertEqual(context(prompt)['operator_question'], question)
        self.assertEqual(context(prompt)['wiki_evidence'], [source])

    def test_failure_is_bounded_unique_and_client_owner_checks_unchanged(self):
        seen = []
        data = messages({'fresh_facts': {}, 'skills': []}, directive={
            'message_id': 'x', 'kind': 'instruction', 'text': 'Observe; do not act.'})
        with self.assertRaisesRegex(ValueError, 'admitted request budget'):
            CURRENT.compact_planner_prompt(data, fits_prompt=lambda text: seen.append(text) or False)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLessEqual(len(seen), 64)
        original = next(n for n in ORIGINAL_TREE.body if isinstance(n, ast.ClassDef)
                        and n.name == 'NativeWorldCognitionModel')
        current = next(n for n in CURRENT_TREE.body if isinstance(n, ast.ClassDef)
                       and n.name == 'NativeWorldCognitionModel')
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(current, include_attributes=False))
        text = ast.unparse(current)
        self.assertIn("'/tokenize'", text)
        self.assertIn('budget_checks > 64', text)
        self.assertIn("budget['prompt_tokens'] <= 512", text)
