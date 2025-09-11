from ocr_web_app import extract_structured_fields

# Sample text from the vessel registration certificate
sample_text = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
SOCIALIST REPUBLIC OF VIETNAM

GIẤY CHỨNG NHẬN ĐĂNG KÝ TÀU CÁ
REGISTRATION CERTIFICATE OF FISHING VESSEL

CHI CỤC THỦY SẢN CÀ MAU
Ca Mau Department Fisheries

Số/No: 230323

Chứng nhận tàu có các thông số dưới đây đã được đăng ký vào Sổ Đăng ký tàu cá quốc gia
Hereby certifies that the fishing vessel with the following specifications has been registered into the Vietnam National Vessel Registration Book

Kiểu tàu/Type of Vessel: LHS43-21/CM
Tổng dung tích, GT/Gross Tonnage: 10.00
Trọng tải toàn phần/Dead weight: 6.31
Chiều dài Lmax m/Length overall: 14.80
Chiều dài thiết kế Lk m/Length: 13.19
Chiều cao mạn D, m/Draught: 1.03
Chiều rộng Bmax m/Breadth overall: 3.23
Chiều rộng thiết kế Bk m/Breadth: 3.18
Chiều chìm d, m/Depth: 0.60

Vật liệu vỏ/Materials: Gỗ/Wood
Nơi và nơi đóng/Year and Place of Build: Cà Mau

Số lượng máy/Number of Engines: 01
Ký hiệu máy/Type of machine: Daewoo D2366
Số máy/Number engines: 4000966
Tổng công suất (KW)/Total power: 169
Công suất (KW)/Power: 169
Nơi và năm chế tạo/Year and place of manufacture: Hàn Quốc

Cảng đăng ký, Hô Gọi/Port Registry: 
Số đăng ký/Number or registry: CM- 08864 -TS

Giấy chứng nhận này có hiệu lực đến/This certificate is valid until: Không thời hạn
Loại đăng ký/Type of registration: DKLD

Cấp tại Cà Mau, ngày 08 tháng 12 năm 2021
Issued at Ca Mau, Date

KT. CHI CỤC TRƯỞNG
PHÓ CHI CỤC TRƯỞNG
"""

# Process the text
print("Processing vessel registration certificate text...")
result = extract_structured_fields(sample_text)

# Print results
print("\nExtracted Fields:")
for field, value in result.items():
    if value:
        print(f"{field}: {value}") 