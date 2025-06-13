import csv

import pandas as pd

class IRC_summary:
    def __init__(self, casename, triples, main_issue, has_main, intro_para):
        input = f'./arg_mining/data/IRC_triples/{casename}_structure_and_embeddings.csv'
        self.output = f'./arg_mining/data/IRC_summaries/{casename}.txt'
        self.df = pd.read_csv(input)
        self.casename = casename
        self.triples = triples
        self.found_majority = False
        self.outcome = ''
        self.conclusion = ''
        self.has_main = has_main
        self.intro_para = intro_para

        # handle Hale's problematic title
        if 'baroness hale' in self.df['judge'].values:
            self.df['judge'] = self.df['judge'].replace('baroness hale', 'hale')
        if 'baroness hale' in self.triples:
            self.triples['hale'] = self.triples.pop('baroness hale')

        self.main_issue = main_issue
        self.summary = []

    def generate_summary(self):
        respondent, appellant = self.get_party_names()
        citation, mj = self.get_citation_majority()

        self.found_majority = len(mj) > 0 and mj != 'NAN' and 'NAN' not in mj

        if self.found_majority:
            self.summarise_case_with_majority(respondent, appellant, citation, mj)
        else:
            self.summarise_case_without_majority(respondent, appellant, citation)

        self.save_summary_to_file()
        return self.summary

    def summarise_case_with_majority(self, respondent, appellant, citation, mj):
        majority_judges = self.format_judge_str(self.format_judge_name(mj))

        self.summary.append(f'In the case of {respondent} v. {appellant} {citation} the majority opinion was '
                            f'delivered by {majority_judges}.\n\n')
        self.summary.append(f'BACKGROUND\n{self.intro_para}\n\n')
        if self.has_main:
            self.summary.append(f'The main issue was: "{self.main_issue}"\n\n')

        for judge in mj:
            if self.triples[judge].get('sub_issues'):
                self.summary.append(f'{self.format_name(judge)} addressed the following issues:\n')
                for idx, issue in enumerate(self.triples[judge]['sub_issues']):
                    self.append_issue(issue, idx+1)
                self.summary.append('\n')

            if self.triples[judge]['main_issue'].get('reasons'):
                if self.has_main:
                    self.summary.append(f'As to the main issue, {self.format_name(judge)} argued:\n')
                else:
                    self.summary.append(f'As the the case in general, {self.format_name(judge)} argued:\n')
                reasons = ' '.join([' '.join(reason) if isinstance(reason, list) else reason for reason in
                                    self.triples[judge]['main_issue']['reasons']])
                self.summary.append(f'{reasons}"\n\n')
                self.summary.append(f'Ultimately {self.format_name(judge)} concluded that "')
                conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion
                                        for conclusion in self.triples[judge]['main_issue']['conclusions']])
                self.summary.append(f'{conclusions}"\n\n')
            else:
                if self.has_main:
                    self.summary.append(f'As to the main issue, {self.format_name(judge)} concluded:\n')
                else:
                    self.summary.append(f'As to the case in general, {self.format_name(judge)} concluded:\n')
                conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion
                                        for conclusion in self.triples[judge]['main_issue']['conclusions']])
                self.summary.append(f'{conclusions}"\n\n')

        self.determine_outcome(mj)
        agree_judges, disagree_judges, agree_with_opinions = self.determine_judge_opinion(mj)

        agree_judge_str = self.format_judge_str(self.format_judge_name(agree_judges))
        disagree_judge_str = self.format_judge_str(self.format_judge_name(disagree_judges))

        if agree_judges:
            self.summary.append(f'{agree_judge_str} agreed with the majority opinion of {majority_judges}.\n')

        if agree_with_opinions:
            for judge in agree_with_opinions:
                self.summary.append(f'{self.format_name(judge)} agreed to the outcome of the case, but was not '
                                    f'of the majority opinion. Instead arguing:\n')
                if self.triples[judge].get('sub_issues'):
                    for idx, issue in enumerate(self.triples[judge]['sub_issues']):
                        self.append_issue(issue, idx + 1)
                    self.summary.append('\n')

                if self.triples[judge]['main_issue'].get('reasons'):
                    if self.has_main:
                        self.summary.append(f'As to the main issue, {self.format_name(judge)} argued:\n')
                    else:
                        self.summary.append(f'As the the case in general, {self.format_name(judge)} argued:\n')
                    reasons = ' '.join([' '.join(reason) if isinstance(reason, list) else reason for reason in
                                        self.triples[judge]['main_issue']['reasons']])
                    self.summary.append(f'{reasons}"\n\n')

                self.summary.append(f'Ultimately {self.format_name(judge)} concluded that "')
                conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion for
                                        conclusion in self.triples[judge]['main_issue']['conclusions']])
                self.summary.append(f'{conclusions}"\n\n')

        if disagree_judges:
            self.summary.append(f'{disagree_judge_str} disagreed with the majority opinion.\n')
            for judge in disagree_judges:
                if self.triples[judge].get('sub_issues'):
                    self.summary.append(f'\n{self.format_name(judge)} argued:\n')
                    for idx, issue in enumerate(self.triples[judge]['sub_issues']):
                        self.append_issue(issue, idx+1)
                if self.has_main:
                    self.summary.append(f'As to the main issue, {self.format_name(judge)} concluded:\n')
                else:
                    self.summary.append(f'As the the case in general, {self.format_name(judge)} concluded:\n')
                conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion
                                        for conclusion in self.triples[judge]['main_issue']['conclusions']])
                self.summary.append(f'{conclusions}"\n\n')

        self.summary.append(f'\n{self.conclusion}')

        print(''.join(map(str, self.summary)))

    def summarise_case_without_majority(self, respondent, appellant, citation):
        judges = list(self.triples.keys())
        opinion_judges = [judge for judge, judge_data in self.triples.items() if judge_data.get('sub_issues')]
        no_opinion_judges = [judge for judge in judges if judge not in opinion_judges]

        opinion_judges_str = self.format_judge_name(opinion_judges)
        opinion_judges_str = self.format_judge_str(opinion_judges_str)

        self.determine_outcome(None)
        # self.determine_judge_agree(opinion_judges, no_opinion_judges)

        if self.has_main:
            self.summary.append(f'In the case of {respondent} v. {appellant} {citation} the main issue was:\n '
                                f'"{self.main_issue}"\n\n')
            self.summary.append(f'{self.intro_para}\n\n')
        else:
            self.summary.append(f'{respondent} v. {appellant} {citation}\n\n')
            self.summary.append(f'{self.intro_para}\n\n')

        if len(opinion_judges) > 1:
            self.summary.append(f'\nOpinions were given by {opinion_judges_str}. ')
        else:
            self.summary.append(f'\nAn opinion was given by {opinion_judges_str}. ')

        for judge in opinion_judges:
            self.summary.append(f'\n{self.format_name(judge)} addressed the following issues:\n')
            for idx, issue in enumerate(self.triples[judge]['sub_issues']):
                self.append_issue(issue, idx+1)
            if self.has_main:
                self.summary.append(f'As to the main issue, {self.format_name(judge)} concluded:\n')
            else:
                self.summary.append(f'As the the case in general, {self.format_name(judge)} concluded:\n')
            conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion
                                    for conclusion in self.triples[judge]['main_issue']['conclusions']])
            self.summary.append(f'{conclusions}"\n\n')

        for judge in no_opinion_judges:
            self.summary.append(f'\n{self.format_name(judge)} concluded that "')
            conclusions = ' '.join([' '.join(conclusion) if isinstance(conclusion, list) else conclusion
                                    for conclusion in self.triples[judge]['main_issue']['conclusions']])
            self.summary.append(f'{conclusions}"\n\n')

        self.summary.append(self.conclusion)

        print(''.join(map(str, self.summary)))

    def get_party_names(self):
        data = pd.read_csv('data/UKHL_corpus/' + self.casename + '.csv')
        return data['text'].iloc[-2], data['text'].iloc[-1]

    def get_citation_majority(self):
        df = pd.read_csv('data/UKHL_corpus/' + self.casename + '.csv')

        df = df[df['role'] == '<new-case>']
        majorities = df['mj'].tolist()
        citations = df['text'].tolist()

        if ',' in majorities[0]:
            majority = majorities[0].split(', ')
            # majority = self.format_judge_name(majority)
        else:
            majority = [majorities[0]]
        return citations[0], majority

    def format_name(self, judge):
        if 'baroness' in judge:
            name = judge[len('baroness'):].lstrip()
            return 'BARONESS ' + name.upper()
        elif 'hale' in judge:
            return 'LADY ' + judge.upper()
        else:
            return 'LORD ' + judge.upper()

    def format_judge_name(self, judges):
        formatted_judges = judges.copy()
        for idx, judge in enumerate(judges):
            formatted_judges[idx] = self.format_name(judge)
        return formatted_judges

    def format_judge_str(self, judges):
        s = ''
        for idx, judge in enumerate(judges):
            if idx == len(judges) - 1:
                s = s + judge
            else:
                s = s + judge + ' and '
        return s

    def format_list(self, text_list):
        if not text_list:
            return ''
        elif len(text_list) == 1:
            return text_list[0]
        else:
            return ' '.join(text_list)

    def append_issue(self, issue, idx):
        issue_text = issue['issue']
        reasons_text = self.format_list(issue['reasons'])
        conclusions_text = self.format_list(issue['conclusions'])

        combined_text = f'({idx}) {issue_text}\n'
        if reasons_text:
            combined_text = combined_text + f'{reasons_text}\n'
        if conclusions_text:
            combined_text = combined_text + f'{conclusions_text}\n'

        # if issue_text and reasons_text and conclusions_text:
        #     combined_text = f'({idx}) "{issue_text}\n{reasons_text}\n{conclusions_text}"'

        self.summary.append(combined_text)

    def determine_judge_opinion(self, mj):
        agree_judges = []
        self.df = self.df[self.df['judge'].notna() & (self.df['judge'] != 'None')]
        judges = self.df['judge'].unique()
        # filter out judges in the majority, so you don't repeat their conclusions
        judges = [judge for judge in judges if judge not in mj]

        for judge in judges:
            judge_df = self.df[self.df['judge'] == judge]
            agreements = judge_df[judge_df['agree'] != 'NONE']['agree'].tolist()
            for agreement in agreements:
                if judge in agree_judges:
                    break

                if '+' in agreement:
                    agreement_to = agreement.split('+')
                    for to in agreement_to:
                        if to in mj:
                            agree_judges.append(judge)
                            break
                else:
                    if agreement in mj:
                        agree_judges.append(judge)
                        break

        disagree_judges = [judge for judge in judges if judge not in agree_judges]
        agree_judges, disagree_judges, agree_opinions = self.check_disagree(agree_judges, disagree_judges)
        return agree_judges, disagree_judges, agree_opinions

    def check_disagree(self, agree, disagree):
        dismiss_sent = ['I would dismiss the appeal', 'should be dismissed', 'I would dismiss',
                        'would dismiss the appeal', 'would therefore dismiss the appeal', 'refuse the appeal',
                        'dismiss the appeal', 'dismiss these appeals', 'this appeal dismissed']
        allow_sent = ['I would allow the appeal', 'would allow the appeal', 'too would allow the appeal',
                      'appeal should be allowed', 'allow the appeal', 'would allow this appeal', 'allow these appeals',
                      'this appeal allowed', 'should proceed to trial']
        agree_sent = ['also make the order', 'make the same order', 'order the same outcome']

        disagree_judges = []
        agree_with_opinions = []
        allow = 0
        dismiss = 0

        for judge in disagree:
            print(f'\n\nChecking {judge}')
            has_opinions = bool(self.triples[judge].get('sub_issues'))
            if self.triples[judge]['main_issue'].get('conclusions'):
                for conclusion in self.triples[judge]['main_issue']['conclusions']:
                    if any(sent in conclusion for sent in allow_sent):
                        print('Found allow statement')
                        allow += 1
                    if any(sent in conclusion for sent in dismiss_sent):
                        print('Found dismiss statement')
                        dismiss += 1
                    if any(sent in conclusion for sent in agree_sent):
                        print('Found order agreement statement')
                        if self.outcome == 'allow':
                            print('Adding to allow')
                            allow += 1
                        elif self.outcome == 'dismiss':
                            print('Adding to dismiss')
                            dismiss += 1

            if has_opinions:
                # in case main_conclusions were mismarked, search all conclusions
                for issue in self.triples[judge]['sub_issues']:
                    for conclusion in issue['conclusions']:
                        if any(sent in conclusion for sent in allow_sent):
                            print(f'{judge} decides to allow appeal')
                            allow += 1
                        elif any(sent in conclusion for sent in dismiss_sent):
                            print(f'{judge} decides to dismiss appeal')
                            dismiss += 1
                        elif any(sent in conclusion for sent in agree_sent):
                            print('Found order agreement statement')
                            if self.outcome == 'allow':
                                print('Adding to allow')
                                allow += 1
                            elif self.outcome == 'dismiss':
                                print('Adding to dismiss')
                                dismiss += 1

            if allow > dismiss:
                if self.outcome == 'allow':
                    print(f'{judge} decides to allow appeal')
                    if has_opinions:
                        agree_with_opinions.append(judge)
                    else:
                        agree.append(judge)
                else:
                    disagree_judges.append(judge)
            elif allow < dismiss:
                print(f'{judge} decides to dismiss appeal')
                if self.outcome == 'dismiss':
                    if has_opinions:
                        agree_with_opinions.append(judge)
                    else:
                        agree.append(judge)
                else:
                    disagree_judges.append(judge)
            else:
                disagree_judges.append(judge)

        return agree, disagree_judges, agree_with_opinions

    def determine_outcome(self, mj):
        dismiss_sent = ['I would dismiss the appeal', 'should be dismissed', 'I would dismiss',
                            'would dismiss the appeal', 'would therefore dismiss the appeal', 'refuse the appeal',
                            'dismiss the appeal']
        allow_sent = ['I would allow the appeal', 'would allow the appeal', 'too would allow the appeal',
                          'appeal should be allowed', 'allow the appeal', 'would allow this appeal']

        allow = 0
        dismiss = 0

        if self.found_majority:
            for judge in mj:
                if self.triples[judge]['main_issue'].get('conclusions'):
                    for conclusion in self.triples[judge]['main_issue']['conclusions']:
                        if isinstance(conclusion, list):
                            conclusion = ' '.join(conclusion)
                        if any(sent in conclusion for sent in allow_sent):
                            print(f'{judge} decides to allow appeal')
                            allow += 1
                        elif any(sent in conclusion for sent in dismiss_sent):
                            print(f'{judge} decides to dismiss appeal')
                            dismiss += 1
                if self.triples[judge].get('sub_issues'):
                    # in case main_conclusions were mismarked, search all majority conclusions
                    for issue in self.triples[judge]['sub_issues']:
                        for conclusion in issue['conclusions']:
                            if isinstance(conclusion, list):
                                conclusion = ' '.join(conclusion)
                            if any(sent in conclusion for sent in allow_sent):
                                print(f'{judge} decides to allow appeal')
                                allow += 1
                            elif any(sent in conclusion for sent in dismiss_sent):
                                print(f'{judge} decides to dismiss appeal')
                                dismiss += 1
        else:
            for judge, judge_data in self.triples.items():
                for conclusion in judge_data['main_issue']['conclusions']:
                    if isinstance(conclusion, list):
                        conclusion = ' '.join(conclusion)
                    if any(sent in conclusion for sent in allow_sent):
                        print(f'{judge} decides to allow appeal')
                        allow += 1
                    elif any(sent in conclusion for sent in dismiss_sent):
                        print(f'{judge} decides to dismiss appeal')
                        dismiss += 1
            if self.triples[judge].get('sub_issues'):
                # in case main_conclusions were mismarked, search all majority conclusions
                for issue in self.triples[judge]['sub_issues']:
                    for conclusion in issue['conclusions']:
                        if isinstance(conclusion, list):
                            conclusion = ' '.join(conclusion)
                        if any(sent in conclusion for sent in allow_sent):
                            print(f'{judge} decides to allow appeal')
                            allow += 1
                        elif any(sent in conclusion for sent in dismiss_sent):
                            print(f'{judge} decides to dismiss appeal')
                            dismiss += 1

        print('\n')
        print(f'For dismiss {dismiss}\nFor allow {allow}')
        print('\n')

        if allow > dismiss:
            self.conclusion = 'The appeal was allowed.'
            self.outcome = 'allow'
        elif allow < dismiss:
            self.conclusion = 'The appeal was dismissed.'
            self.outcome = 'dismiss'
        else:
            print('\n\nuh oh we found an anomaly\n\n')

    def save_summary_to_file(self):
        with open(self.output, 'w', encoding='utf-8') as file:
            for line in self.summary:
                file.write(line)







