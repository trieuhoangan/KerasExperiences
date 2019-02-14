import pymysql.cursors
from nltk.tokenize import sent_tokenize
import re
    
dbname = 'AliceII'  

def saveContentIntoNewspaperTable(_host,_user,_password,_db,title,content):
    connection = pymysql.connect(host = _host,user = _user, password = _password, db = _db, charset="utf8",cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `title` FROM `newspaper` WHERE `title`=%s"
            cursor.execute(sql, (title))
            result = cursor.fetchone()
            if bool(result)==False:
                sql = "INSERT INTO `newspaper` (`title`, `content`,`sentences`) VALUES (%s, %s, %s)"   
                sentences = sent_tokenize(content)
                numberOfSentence = len(sentences)
                if numberOfSentence >10:
                    cursor.execute(sql, (title,content,numberOfSentence)) 
                    print("saving document......")
        connection.commit()
    finally:
        connection.close()    


def save_content_to_file(content,title,folder):
    filename = '%s/%s.txt'%(folder,title)
    f = open(filename,'w',encoding='utf-8')
    f.write(content)


def normalizeContent(content):
    cleanr = re.compile('<.*?>',flags=re.DOTALL)
    clean_spe_char = re.compile('[\\~#%&*{}/:<>?|\"-]')
    vb = re.sub(cleanr,'',content)
    vb = vb.replace('\n','')
    vb = vb.replace('\r','')
    vb = re.sub(clean_spe_char,'',vb)
    return vb


def normalizeTitle(title):
    valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
    title = re.sub(valid_file_name_character,'',title)
    title = title.replace('\n','_')
    title = title.replace(' ','_')
    title = title.strip()
    return title

def isViolent(content,_host,_user,_password,_db,_table):
    connection = pymysql.connect(host = _host,user = _user, password = _password, db = _db, charset="utf8",cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `violentWord` FROM %s"% _table
            cursor.execute(sql) 
            results = cursor.fetchall()
            for result in results:
                if result.get('violentWord') in content:
                    return True
    finally:
        connection.close()    
    
    return False

def getContentList(sql):
    texts = []
    connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db=dbname,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            for result in results:
                # if(int(result.get('label'))>0):
                content = str(result.get('content'))
                content.encode('ascii', 'replace')
                texts.append(content)
        connection.commit()
    finally:
        connection.close()
    return texts    

def getLabelList(sql):
    labels = []
    connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db=dbname,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            for result in results:
                if(result.get('label') != None):
                    label_id = result.get('label')
                    labels.append(int(label_id))
        connection.commit()
    finally:
        connection.close()
    return labels        


    