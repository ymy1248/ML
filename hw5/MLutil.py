import pickle
import os.path
import ast
import numpy as np
from collections import OrderedDict

def send_email(user, pwd, recipient, subject, body):
	import smtplib

	gmail_user = user
	gmail_pwd = pwd
	FROM = user
	TO = recipient if type(recipient) is list else [recipient]
	SUBJECT = subject
	TEXT = body

	# Prepare actual self.message
	message = """From: %s\nTo: %s\nSubject: %s\n\n%s
	""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
	try:
		server = smtplib.SMTP("smtp.gmail.com", 587)
		server.ehlo()
		server.starttls()
		server.ehlo()
		server.login(gmail_user, gmail_pwd)
		server.sendmail(FROM, TO, message)
		server.close()
		print("successfully sent the mail")
	except Exception as e:
		print(e)
		print("failed to send mail")


class Storer:
    def __init__ (self, file_name):
        self.file_name = file_name
        self.message = ''
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.dict = pickle.load(f)
            f.close()
        else:
            self.dict = OrderedDict()

    def check_model(self, conf):
        if conf in self.dict:
            self.message += conf + '\n already in ML storer!\n'
            for key, value in self.dict[conf].items():
                self.message += key + ':' + '{0:.4f}'.format(np.max(value)) + '\n'
        return conf in self.dict

    def store (self, conf, value):
        self.dict[conf] = value
        self.message += conf + '\n'
        for key, sub_value in value.items():
            self.message += key + ':' + '{0:.4f}'.format(np.max(sub_value)) + '\n'
        self.dict = OrderedDict(sorted(self.dict.items(), key = lambda t:t[0]))

    def show_dict (self):
        print(self.dict)

    def show_key (self):
        for key in self.dict.keys():
            print(key)

    def show_record_item(self):
        value = next(iter(self.dict.values()))
        print('You can chose:')
        for key in value.keys():
            print(key)

    def show_all_infor(self):
        self.dict = OrderedDict(sorted(self.dict.items(), key = lambda t:t[0]))
        for key, value in self.dict.items():
            print(key)
            for sub_key, sub_value in value.items():
                if sub_key == 'loss' or sub_key == 'val_loss':
                    print(sub_key + ': {0:.4f}'.format(np.min(sub_value)))
                else:
                    print(sub_key + ': {0:.4f}'.format(np.max(sub_value)))
            print('-------------------------------------------')

    def reorder_val(self):
        for key, value in self.dict.items():
            new_value = OrderedDict()
            new_value['loss'] = value['loss']
            new_value['acc'] = value['acc']
            new_value['f1_score'] = value['f1_score']
            new_value['val_loss'] = value['val_loss']
            new_value['val_acc'] = value['val_acc']
            new_value['val_f1_scores'] = value['val_f1_scores']
            self.dict[key] = new_value

    def rename_key(self):
        reg_dict = OrderedDict()
        for key, value in self.dict.items():
            sub_key = ast.literal_eval(key)
            #rnn_num = len(sub_key[1])
            #dnn_num = len(sub_key[3])
            #sub_key.insert(1, rnn_num)
            #sub_key.insert(kk

        self.dict = reg_dict

    def close(self):
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.dict, f) 
        f.close()
        send_email("yemengyuan0405", "fei_EFH_214fd", "carlosyex@gmail.com", "r04921094@ntu.edu.tw", self.message)

if __name__ == '__main__':
    storer = Storer('rnn_model')
    #storer.store('zapple', 2345)
    #storer.store('appe', 2345)
    #storer.store('paple', 2345)
    #storer.store('pale', 2345)
    storer.show_all_infor()
