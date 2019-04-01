import tika
import ast
import dill
from selenium import webdriver
import justext
import json
import scholarly
import traceback
import time

phantomjs_exec_path='/usr/local/bin/phantomjs'
timelimit=20
options = webdriver.ChromeOptions()

def quitdriver(driver):
    try:
        # driver.service.process.send_signal(signal.SIGTERM)
        driver.quit()
        print('driver quit successfully!')
    except:
        error = traceback.format_exc()
        print('error quitting',error)

    return



def featchcontent(url):
    driver=''
    try:
        driver = webdriver.Chrome(chrome_options=options)
        driver.set_page_load_timeout(timelimit)  # seconds
        driver.get(url)
        time.sleep(2)    # pause
        paragraphs = justext.justext(driver.page_source,[])
        all_texts=[]
        for paragraph in paragraphs:
            txt=paragraph.text
            all_texts.append(txt)

        quitdriver(driver)
        return all_texts

    except:
        error = traceback.format_exc()
        print(error)
        if driver!='':
            quitdriver(driver)
        return []


mapping=open('400_mapping.txt','r').readlines()
fileout=open('remaining_400_final_json.json','w')

# remaining_fina_json={}
# for i,line in enumerate(mapping):
#     if i<=270:
#         continue
#     print(i)
#     content=''
#     url=line.split()[0]
#     label=line.split()[1]
#     if 'academic.oup.com' in url:
#         content=featchcontent('https://'+url)
#         content='\n'.join(content)
#     remaining_fina_json['url']=url
#     remaining_fina_json['class']=label
#     remaining_fina_json['content']=content
#     fileout.write(json.dumps(remaining_fina_json)+'\n')
#     fileout.flush()

final_json=[]
seen=set()
for i in range(1,9):
    file=open('remaining_400_final_json_'+str(i)+'.json','r')
    for line in file:
        jsn=json.loads(line)
        if 'academic.oup.com' in jsn['url']:
            if jsn['url'] in seen:
                continue
            final_json.append(jsn)
            seen.add(jsn['url'])

fileout.write(json.dumps(final_json))



