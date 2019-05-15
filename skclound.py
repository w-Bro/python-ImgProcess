"""
https://skcloudpro.com/
每日签到领流量
"""

from selenium import webdriver

if __name__ == '__main__':
    url = 'https://skcloudpro.com/auth/login'
    username = 'aptxconan48691327@gmail.com'
    password = 'conan4869'
    
    driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
    
    driver.get(url)
    
    email_box = driver.find_element_by_xpath("//input[@id='email']")
    email_box.send_keys(username)
    passwd_box = driver.find_element_by_xpath("//input[@id='passwd']")
    passwd_box.send_keys(password)
    
    driver.find_element_by_xpath("//button[@id='login']").click()
    
    import time
    time.sleep(3)
    
    try:
        driver.find_element_by_xpath("//button[@id='checkin']").click()
    except:
        print('今日已签到')
    finally:
        time.sleep(3)
        driver.quit()