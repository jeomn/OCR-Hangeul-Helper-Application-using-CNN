from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pymongo

driver = webdriver.Chrome('C:\pythonlib\chromedriver_win32\chromedriver.exe')
connection = pymongo.MongoClient('localhost', 27017)
db=connection.for_graduation
collection=db.store

def Crawling(subject, first_page, last_page):
    for i in range(first_page, last_page):
        driver.get("https://map.naver.com/?query=" + subject + '&page=' + str(i))
        time.sleep(5)
        print('\n'+str(i) + ' 페이지')
        source = driver.page_source
        bs = BeautifulSoup(source,'lxml')
        entire = bs.find('ul',class_='lst_site')
        li_lists = entire.find_all('li')

        for infor in li_lists:
            if infor.find('div', class_='lsnx'):
                infor_name = infor.find('dt')
                infor_name = infor_name.find('a').text.strip()

                infor_addr = infor.find('dd', class_='addr')
                infor_addr_child = infor_addr.findChildren(text=True)
                infor_addr = infor_addr_child[0].strip()

                #infor_tel = infor.find('dd',class_='tel').text.strip()
                if infor.find('dd', class_='tel') is None:
                    infor_tel = None
                else:
                    infor_tel = infor.find('dd',class_='tel').text.strip()

                infor_cate = infor.find('dd',class_='cate').text.strip()
                shop_info = {'Store_Name':infor_name,
                             'Store_Addr':infor_addr,
                             'Store_Tel': infor_tel,
                             'Store_Cate': infor_cate}
                aa = collection.find_one({"$and":[{"Store_Name": infor_name}, {"Store_Addr": infor_addr}]})
                if aa is not None:
                    print("중복"+str(aa))
                    print(aa['_id'])
                else :
                    collection.save(shop_info)
                    print(shop_info)

Crawling('정왕 쇼핑,유통 > 신발', 1,3)

print("------finish------")