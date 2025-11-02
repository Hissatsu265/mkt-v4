import requests
from typing import List, Tuple, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class AudioKeywordExtractor:
    def __init__(self, auth_token: str):
        """
        Khởi tạo extractor với token xác thực
        
        Args:
            auth_token: Bearer token để xác thực API
        """
        self.auth_token = auth_token
        self.subtitle_url = "https://dev.shohanursobuj.online/api/v1/marketing-video/generate-subtitles"
        self.keyword_url = "https://dev.shohanursobuj.online/api/v1/marketing-video/extract-keywords"
    
    def generate_subtitles(self, audio_path: str, language: str = "") -> Dict[str, Any]:
        """
        Gọi API để tạo subtitles từ file audio
        
        Args:
            audio_path: Đường dẫn đến file audio
            language: Ngôn ngữ (để trống để auto-detect)
            
        Returns:
            Response JSON từ API
        """
        params = {
            "format": "json",
            "language": language
        }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "accept": "application/json"
        }
        
        # Lấy tên file từ đường dẫn
        filename = audio_path.split("/")[-1]
        
        with open(audio_path, "rb") as f:
            files = {
                "audio_file": (filename, f, "audio/wav")
            }
            
            response = requests.post(
                self.subtitle_url, 
                headers=headers, 
                params=params, 
                files=files
            )
        
        if response.status_code != 200:
            raise Exception(f"❌ Lỗi khi tạo subtitles: {response.status_code} - {response.text}")
        
        return response.json()
    
    def extract_keywords(self, text: str) -> Dict[str, Any]:

        try:
            data = {"text": text}
            response = requests.post(self.keyword_url, data=data)
            
            if response.status_code != 200:
                print(f"⚠️ Không thể trích xuất keywords: {response.status_code} - {response.text}")
                return {"keywords": []}
            
            return response.json()
        except Exception as e:
            print(f"⚠️ Lỗi khi gọi API keywords: {e}")
            return {"keywords": []}
    
    def find_keyword_timeline(
        self, 
        keyword: str, 
        segments: List[Dict]
    ) -> List[Tuple[float, float]]:

        timeline = []
        keyword_lower = keyword.lower()
        
        for segment in segments:
            words = segment.get('words', [])
            
            # Xử lý keyword đơn
            if ' ' not in keyword:
                for word in words:
                    word_text = word['word'].strip().lower().rstrip('.,!?;:')
                    if word_text == keyword_lower:
                        timeline.append((word['start'], word['end']))
            
            # Xử lý keyword nhiều từ (phrase)
            else:
                keyword_words = keyword_lower.split()
                for i in range(len(words) - len(keyword_words) + 1):
                    # Kiểm tra chuỗi từ liên tiếp
                    match = True
                    for j, kw in enumerate(keyword_words):
                        word_text = words[i + j]['word'].strip().lower().rstrip('.,!?;:')
                        if word_text != kw:
                            match = False
                            break
                    
                    if match:
                        start_time = words[i]['start']
                        end_time = words[i + len(keyword_words) - 1]['end']
                        timeline.append((start_time, end_time))
        
        return timeline
    
    def process_audio(
        self, 
        audio_path: str, 
        language: str = ""
    ) -> Tuple[List[str], List[List[float]], List[List[float]]]:
        try:
            print("📝 Đang tạo subtitles từ audio...")
            subtitle_data = self.generate_subtitles(audio_path, language)
            
            segments = subtitle_data.get('segments', [])
            full_text = ' '.join([seg['text'] for seg in segments])
            
            print(f"✅ Đã tạo subtitles: {subtitle_data.get('total_segments')} segments")
            print(f"📄 Text: {full_text}\n")
            
            print("🔍 Đang trích xuất keywords...")
            keyword_data = self.extract_keywords(full_text)
            keywords = keyword_data.get('keywords', [])

            if not keywords:
                print("⚠️ Không tìm thấy keywords nào")
                return [], [], []
            
            start_times_list = []
            end_times_list = []
            for keyword in keywords:
                timeline = self.find_keyword_timeline(keyword, segments)
                
                if timeline:
                    start_times = [t[0] for t in timeline]
                    end_times = [t[1] for t in timeline]
                    start_times_list.append(start_times)
                    end_times_list.append(end_times)
                else:
                    start_times_list.append([])
                    end_times_list.append([])
            
            return keywords, start_times_list, end_times_list
        
        except Exception as e:
            print(f"❌ Lỗi trong process_audio: {e}")
            return [], [], []



def process_keywordfromaudi(audio_path):
    AUDIO_FILE =audio_path
    extractor = AudioKeywordExtractor(os.getenv("AUTH_TOKEN"))
    
    try:
        keywords, start_times, end_times = extractor.process_audio(AUDIO_FILE) 
        combined = []
        for kw, starts, ends in zip(keywords, start_times, end_times):
            # Chỉ thêm khi cả start và end có dữ liệu
            if starts and ends:
                # Số lần xuất hiện có thể khác nhau → lấy theo min độ dài
                for s, e in zip(starts, ends):
                    combined.append((kw, s, e))

        # Bước 2: sắp xếp giảm dần theo start time
        combined.sort(key=lambda x: x[1])

        # Bước 3: tách lại thành 3 mảng 1 chiều
        sorted_keywords = [x[0] for x in combined]
        sorted_starts = [x[1] for x in combined]
        sorted_ends = [x[2] for x in combined]

        # print("📋 Kết quả:")
        # print("keywords =", sorted_keywords)
        # print("start_times =", sorted_starts)
        # print("end_times =", sorted_ends)  
        return   sorted_keywords, sorted_starts,sorted_ends
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return [],[],[]
# k,s,e= process_keywordfromaudi("/home/toan/marketing-video-ai./download_audios/1a44673228584f8b9877bcf6ff8bec88.mp3")
# print(k)
# print(s)
# print(e)