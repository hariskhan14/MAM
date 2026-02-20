import argparse
import sys


def safe_import(module_name, attr=None):
    try:
        mod = __import__(module_name, fromlist=['*'])
        if attr:
            return getattr(mod, attr)
        return mod
    except Exception as e:
        print(f'Warning: could not import {module_name}{("." + attr) if attr else ""}: {e}')


modality_selection = safe_import('pipeline.modality_selection', 'modality_selection')
type_classification = safe_import('pipeline.type_classification', 'type_classification')
generate_role = safe_import('pipeline.role_generation', 'generate_role')
roles_meeting = safe_import('pipeline.meeting', 'roles_meeting')
final_diagnosis = safe_import('pipeline.diagnosis', 'final_diagnosis')
review_all = safe_import('pipeline.review', 'review_all')
memory = safe_import('pipeline.memory', 'memory')
web_search_check = safe_import('pipeline.web_search_check', 'WebSearchCheck')


def run_step_9(question, file_name):
    print('Running end-to-end pipeline (step 9)')

    modality = None
    if modality_selection:
        modality = modality_selection(question, file_name)
    else:
        print('modality_selection not available; defaulting to text')
        modality = 'text'

    print(f'Modality: {modality}')

    type_name = None
    if type_classification:
        try:
            type_name = type_classification(modality, question, file_name)
        except Exception as e:
            print('type_classification failed:', e)

    if not type_name:
        type_name = 'general'

    print(f'Type: {type_name}')

    roles_generated = None
    if generate_role:
        try:
            roles_generated = generate_role(type_name, modality, question, file_name)
        except Exception as e:
            print('generate_role failed:', e)

    meeting_record = ''
    if roles_meeting and roles_generated is not None:
        try:
            meeting_record = roles_meeting(question, file_name, modality, type_name, roles_generated, '')
        except Exception as e:
            print('roles_meeting failed:', e)

    diagnosis = None
    if final_diagnosis:
        try:
            diagnosis = final_diagnosis(question, file_name, modality, type_name, meeting_record)
        except Exception as e:
            print('final_diagnosis failed:', e)

    if review_all and diagnosis is not None:
        try:
            review = review_all(question, file_name, modality, type_name, diagnosis)
            print('Review result:', review)
        except Exception as e:
            print('review_all failed:', e)

    if memory and diagnosis is not None:
        try:
            memory(1, question, file_name, modality, diagnosis)
        except Exception as e:
            print('memory save failed:', e)

    if web_search_check:
        try:
            search_summary = web_search_check(question, file_name, modality)
            print('Web search summary:', search_summary)
        except Exception as e:
            print('web_search_check failed:', e)

    print('\n--- End of pipeline ---')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_id', type=int, default=9, help='Pipeline step to run (9 = end-to-end)')
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--file_name', type=str, default='')
    args = parser.parse_args()

    if args.step_id == 9:
        run_step_9(args.question, args.file_name)
    else:
        print(f'Step {args.step_id} is not implemented in this lightweight driver.')


if __name__ == '__main__':
    main()