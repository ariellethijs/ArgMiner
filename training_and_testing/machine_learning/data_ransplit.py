
"""


@author: rozano aulia
"""

import random
import csv

caseList = ['1.19', '1.63', '1.68', 'N/A',
'1.05', '1.02', '1.04', '1.35', '1.39',
'1.38', '1.42', '1.34', '1.11', '1.15',
'1.26', '1.28', '1.57', '1.43', '1.55',
'2.13', '2.18', '2.3', '2.35', '2.34',
'2.26', '2.24', '2.29', '2.21', '2.23',
'2.45', '2.47', '2.41', '3.18', '3.21',
'3.22', '3.07', '3.1', '3.08', '3.02',
'3.44', '3.41', '3.31', '3.32',
'3.15', '3.14', '3.28']

def data_ransplit():
    ransplit = random.sample(caseList, 6)

    fieldnames = ['case_id', 'sent_id', 'align', 'agree', 'outcome', 'loc1', 'loc2', 'loc3',
                  'loc4', 'loc5', 'loc6', 'HGloc1', 'HGloc2', 'HGloc3', 'HGloc4', 'HGloc5', 'HGloc6', 'sentlen',
                  'HGsentlen', 'quoteblock', 'inline_q', 'rhet', 'tfidf_max', 'tfidf_top20', 'tfidf_HGavg', 'aspect',
                  'modal',
                  'voice', 'negation', 'tense', 'case entities', 'legal entities', 'enamex', 'rhet_target', 'wordlist',
                  'past tense',
                  'case name entity', 'loc ent', 'org ent', 'date ent', 'person ent', 'fac_ent', 'norp_ent', 'gpe_ent',
                  'event_ent', 'law_ent', 'time_ent',
                  'work_of_art_ent', 'ordinal_ent', 'cardinal_ent', 'money_ent', 'percent_ent', 'product_ent',
                  'quantity_ent', 'judgename', 'rhet label',
                  'cp tense', 'cp modal', 'cp pos bool', 'cp dep bool', 'cp dep count', 'cp pos count', 'cp dep',
                  'cp tag', 'cp negative',
                  'cp stop', 'cp voice', 'cp second pos', 'cp second dep', 'cp second tag', 'cp second stop']

    with open("./data/MLdata.csv", "r") as infile, \
    open("./data/MLdata_train.csv", "w", newline="") as train_outfile, \
    open("./data/MLdata_test.csv", "w", newline="") as test_outfile:
        reader = csv.DictReader(infile)

        train_writer = csv.DictWriter(train_outfile, fieldnames=fieldnames)
        train_writer.writeheader()

        test_writer = csv.DictWriter(test_outfile, fieldnames=fieldnames)
        test_writer.writeheader()

        for row in reader:
            if row['case_id'] in ransplit:
                test_writer.writerow({'case_id': row['case_id'], 'sent_id': row['sent_id'], 'align': row['align'], 'agree': row['agree'],
                'outcome': row['outcome'], 'loc1': row['loc1'], 'loc2': row['loc2'], 'loc3': row['loc3'], 'loc4': row['loc4'],
                'loc5': row['loc5'], 'loc6': row['loc6'], 'HGloc1': row['HGloc1'], 'HGloc2': row['HGloc2'], 'HGloc3': row['HGloc3'], 'HGloc4': row['HGloc4'],
                'HGloc5': row['HGloc5'], 'HGloc6': row['HGloc6'], 'sentlen': row['sentlen'], 'HGsentlen': row['HGsentlen'], 'quoteblock': row['quoteblock'], 'inline_q': row['inline_q'],
                'rhet': row['rhet'], 'tfidf_max': row['tfidf_max'], 'tfidf_top20': row['tfidf_top20'], 'tfidf_HGavg': row['tfidf_HGavg'], 'aspect': row['aspect'],
                'modal': row['modal'], 'voice': row['voice'], 'negation': row['negation'], 'tense': row['tense'], 'case entities': row['case entities'],
                'legal entities': row['legal entities'], 'enamex': row['enamex'], 'rhet_target': row['rhet_target'], 'wordlist': row['wordlist'],
                'past tense': row['past tense'],'case name entity': row['case name entity'], 'loc ent': row['loc ent'], 'org ent': row['org ent'], 'date ent': row['date ent'],
                'person ent': row['person ent'],'fac_ent': row['fac_ent'], 'norp_ent': row['norp_ent'], 'gpe_ent': row['gpe_ent'], 'event_ent': row['event_ent'], 'law_ent': row['law_ent'],
                'time_ent': row['time_ent'], 'work_of_art_ent': row['work_of_art_ent'], 'ordinal_ent': row['ordinal_ent'], 'cardinal_ent': row['cardinal_ent'], 'money_ent': row['money_ent'],
                'percent_ent': row['percent_ent'],'product_ent': row['product_ent'], 'quantity_ent': row['quantity_ent'], 'judgename': row['judgename'], 'rhet label': row['rhet label'],
                'cp tense': row['cp tense'], 'cp modal': row['cp modal'], 'cp pos bool': row['cp pos bool'], 'cp dep bool': row['cp dep bool'], 'cp dep count': row['cp dep count'],
                'cp pos count': row['cp pos count'], 'cp dep': row['cp dep'], 'cp tag': row['cp tag'], 'cp negative': row['cp negative'],  'cp stop': row['cp stop'], 'cp voice': row['cp voice'],
                'cp second pos': row['cp second pos'], 'cp second dep': row['cp second dep'], 'cp second tag': row['cp second tag'], 'cp second stop': row['cp second stop']})
            else:
                train_writer.writerow({'case_id': row['case_id'], 'sent_id': row['sent_id'], 'align': row['align'], 'agree': row['agree'],
                'outcome': row['outcome'], 'loc1': row['loc1'], 'loc2': row['loc2'], 'loc3': row['loc3'], 'loc4': row['loc4'],
                'loc5': row['loc5'], 'loc6': row['loc6'], 'HGloc1': row['HGloc1'], 'HGloc2': row['HGloc2'], 'HGloc3': row['HGloc3'], 'HGloc4': row['HGloc4'],
                'HGloc5': row['HGloc5'], 'HGloc6': row['HGloc6'], 'sentlen': row['sentlen'], 'HGsentlen': row['HGsentlen'], 'quoteblock': row['quoteblock'], 'inline_q': row['inline_q'],
                'rhet': row['rhet'], 'tfidf_max': row['tfidf_max'], 'tfidf_top20': row['tfidf_top20'], 'tfidf_HGavg': row['tfidf_HGavg'], 'aspect': row['aspect'],
                'modal': row['modal'], 'voice': row['voice'], 'negation': row['negation'], 'tense': row['tense'], 'case entities': row['case entities'],
                'legal entities': row['legal entities'], 'enamex': row['enamex'], 'rhet_target': row['rhet_target'], 'wordlist': row['wordlist'],
                'past tense': row['past tense'],'case name entity': row['case name entity'], 'loc ent': row['loc ent'], 'org ent': row['org ent'], 'date ent': row['date ent'],
                'person ent': row['person ent'],'fac_ent': row['fac_ent'], 'norp_ent': row['norp_ent'], 'gpe_ent': row['gpe_ent'], 'event_ent': row['event_ent'], 'law_ent': row['law_ent'],
                'time_ent': row['time_ent'], 'work_of_art_ent': row['work_of_art_ent'], 'ordinal_ent': row['ordinal_ent'], 'cardinal_ent': row['cardinal_ent'], 'money_ent': row['money_ent'],
                'percent_ent': row['percent_ent'],'product_ent': row['product_ent'], 'quantity_ent': row['quantity_ent'], 'judgename': row['judgename'], 'rhet label': row['rhet label'],
                'cp tense': row['cp tense'], 'cp modal': row['cp modal'], 'cp pos bool': row['cp pos bool'], 'cp dep bool': row['cp dep bool'], 'cp dep count': row['cp dep count'],
                'cp pos count': row['cp pos count'], 'cp dep': row['cp dep'], 'cp tag': row['cp tag'], 'cp negative': row['cp negative'], 'cp stop': row['cp stop'], 'cp voice': row['cp voice'],
                'cp second pos': row['cp second pos'], 'cp second dep': row['cp second dep'], 'cp second tag': row['cp second tag'], 'cp second stop': row['cp second stop']})

def remove_testsplit():
    test_cases = []
    with open("./data/MLdata_test.csv", "r", ) as test_infile:
        reader = csv.DictReader(test_infile)

        for row in reader:
            if row["case_id"] not in test_cases:
                test_cases.append(row["case_id"])
        exit()
    with open("./data/MLdata68.csv", "r") as infile, \
    open("./data/MLdata68_train.csv", "w", newline="") as train68_outfile:
        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames()
        writer = csv.DictWriter(train68_outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if row['case_id'] not in test_cases:
                writer.writerow(row)

#remove_testsplit()
data_ransplit()