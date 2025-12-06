import pandas as pd
import time
import re
import numpy as np
from scipy import stats
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging

# 这个爬虫在Windrows下跑的，但是Linux下把用户目录改改也能用

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_NAME = 'dat.csv'

if __name__ == "__main__":
    # 启动Chrome
    chrome_options = Options()

    # 用户目录 用于登陆Google 爬到的结果会好很多
    # chrome_options.add_argument(r"--user-data-dir=D:\Profile")
    # chrome_options.add_argument("--profile-directory=Default")

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
    driver = webdriver.Chrome(options=chrome_options)
    logger.info("Chrome is Running")

    # driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    #     "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    # })

    def close_chrome():
        if driver:
            driver.quit()
            logger.info("Chrome closed.")

    def search_car_specs(brand, model, engine_model, year):
        try:
            search_query = f"{brand} {model} {year} engine specs horsepower seats"
            logger.info(f"Searching {search_query}")

            url = f"https://www.google.com/search?q={search_query}"
            driver.get(url)
            time.sleep(2) # 等加载

            page_text = driver.page_source.lower()
            
            engine_cc = extract_engine_cc(engine_model, page_text)
            max_power = extract_max_power(page_text)
            seats = extract_seats(page_text)
            
            return engine_cc, max_power, seats
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return None, None, None
    
    def extract_engine_cc(engine_model, page_text):
        try:
            if pd.notna(engine_model) and str(engine_model).strip() != '':
                engine_str = str(engine_model).lower()
                
                l_pattern = r'(\d+\.?\d*)\s*l' # (n)l
                match = re.search(l_pattern, engine_str)
                if match:
                    liters = float(match.group(1))
                    cc = int(liters * 1000)
                    logger.info(f"From engne_model: {cc}CC")
                    return cc
                
                match = re.search(l_pattern, page_text)
                if match:
                    liters = float(match.group(1))
                    cc = int(liters * 1000)
                    logger.info(f"From page: {cc}CC")
                    return cc
            
            return None
        except Exception as e:
            logger.error(f"Error engine_cc: {e}")
            return None
    
    def extract_max_power(page_text):
        try:
            # (n)hp|bhp
            patterns = [
                r'(\d+)\s*(?:hp|bhp)',
                r'(?:power|horsepower)[:\s]+(\d+)\s*(?:hp|bhp)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    power = np.array([int(m) for m in matches])
                    # 如果找到的不止一个，那就取众数/最大值
                    ret = None
                    if(power.size > 1):
                        ret = stats.mode(power)[0]
                    else:
                        ret = power[0]
                    logger.info(f"Max Power: {ret}bhp")
                    return ret
                    # return np.max(power)
            
            return None
        except Exception as e:
            logger.error(f"Error max_power: {e}")
            return None
    
    def extract_seats(page_text):
        try:
            # (n)seats
            patterns = [
                r'(\d+)\s*seats',
                r'seating[:\s]*(\d+)',
                r'passengers[:\s]*(\d+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    seats = int(matches[0])
                    # 只要座位数在[1,10] 找不到就默认5座位
                    if not (1 <= seats <= 10):
                        seats = 5

                    logger.info(f"Seats: {seats}")
                    return seats

            return None
        except Exception as e:
            logger.error(f"Error seats:: {e}")
            return None
    
    def save(row_idx, engine, max_power, seats):
        try:
            if engine is not None:
                df.at[row_idx, 'Engine'] = engine
            if max_power is not None:
                df.at[row_idx, 'Max Power'] = max_power
            if seats is not None:
                df.at[row_idx, 'Seats'] = seats

            df.to_csv(FILE_NAME, index=False)
            logger.info(f"Saved {row_idx} line.")
            
        except Exception as e:
            logger.error(f"Error save file: {e}")

    df = pd.read_csv(FILE_NAME)

    # 从第一个要爬的三个类别都空的行开始爬
    start_line = -1
    target_cols = ['Engine', 'Max Power', 'Seats']
    for idx, row in df.iterrows():
        if all(pd.isna(row[col]) or str(row[col]).strip() == '' for col in target_cols):
            logger.info(f"Start from {idx} line")
            start_line = idx
            break
    
    if start_line == -1:
        logger.info("All line finished.")
        exit()
    
    try:
        for idx in range(start_line, len(df)):
            target_cols = ['Engine', 'Max Power', 'Seats']
            if not all(pd.isna(df.iloc[idx][col]) or str(df.iloc[idx][col]).strip() == '' for col in target_cols):
                logger.info(f"Skip {idx} line")
                continue

            brand = df.iloc[idx]['Brand']
            model = df.iloc[idx]['Model']
            engine_model = df.iloc[idx]['engine model']
            year = df.iloc[idx].get('Year', '')
            
            logger.info(f"On line {idx}: {brand} {model} ({year})")

            engine, max_power, seats = search_car_specs(brand, model, engine_model, year)

            save(idx, engine, max_power, seats)
            
            time.sleep(3) # 怕被封
            
        logger.info("Finished successfully.")
        exit()
            
    except Exception as e:
        logger.error(f"Crawler error: {e}")
        exit(-1)
    finally:
        close_chrome()
    
