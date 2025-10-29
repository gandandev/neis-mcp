#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from fastmcp import FastMCP


class NeisAPIError(RuntimeError):
    """Raised when the NEIS API returns an unexpected response."""


class MissingNeisAPIKeyError(NeisAPIError):
    """Raised when the NEIS API key is missing."""


TIMETABLE_ENDPOINTS: Dict[str, str] = {
    "elementary": "elsTimetable",
    "middle": "misTimetable",
    "high": "hisTimetable",
    "special": "spsTimetable",
}

OFFICE_OF_EDUCATION_CODES: Dict[str, str] = {
    "서울특별시교육청": "B10",
    "부산광역시교육청": "C10",
    "대구광역시교육청": "D10",
    "인천광역시교육청": "E10",
    "광주광역시교육청": "F10",
    "대전광역시교육청": "G10",
    "울산광역시교육청": "H10",
    "세종특별자치시교육청": "I10",
    "경기도교육청": "J10",
    "강원특별자치도교육청": "K10",
    "충청북도교육청": "M10",
    "충청남도교육청": "N10",
    "전라북도교육청": "P10",
    "전라남도교육청": "Q10",
    "경상북도교육청": "R10",
    "경상남도교육청": "S10",
    "제주특별자치도교육청": "T10",
}


@dataclass(slots=True)
class NeisClient:
    base_url: str = "https://open.neis.go.kr/hub"
    api_key: Optional[str] = None
    timeout: int = 10

    def request(self, endpoint: str, **params: Any) -> List[Dict[str, Any]]:
        api_key = self.api_key or os.environ.get("NEIS_API_KEY")
        if not api_key:
            raise MissingNeisAPIKeyError(
                "NEIS API key is required. Set the NEIS_API_KEY environment variable."
            )

        query: Dict[str, Any] = {"KEY": api_key, "Type": "json"}
        query.update({k: v for k, v in params.items() if v is not None})

        url = f"{self.base_url.rstrip('/')}/{endpoint}?{urlencode(query)}"
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = response.read()
        except HTTPError as http_error:
            raise NeisAPIError(
                f"NEIS API HTTP error {http_error.code}: {http_error.reason}"
            ) from http_error
        except URLError as url_error:
            raise NeisAPIError(
                f"Failed to reach NEIS API: {url_error.reason}"
            ) from url_error

        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as json_error:
            raise NeisAPIError(
                "Failed to parse NEIS API response as JSON."
            ) from json_error

        return self._extract_rows(endpoint, data)

    @staticmethod
    def _extract_rows(endpoint: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        dataset = data.get(endpoint)
        if not dataset:
            result = data.get("RESULT")
            if isinstance(result, dict):
                code = result.get("CODE", "UNKNOWN")
                message = result.get("MESSAGE", "No message provided.")
                raise NeisAPIError(f"NEIS API error {code}: {message}")
            raise NeisAPIError("NEIS API returned an unexpected response structure.")

        head = dataset[0].get("head", []) if dataset else []
        result_info = next(
            (
                entry.get("RESULT")
                for entry in head
                if isinstance(entry, dict) and "RESULT" in entry
            ),
            None,
        )

        if result_info:
            code = result_info.get("CODE")
            message = result_info.get("MESSAGE", "")
            if code == "INFO-000":
                pass
            elif code == "INFO-200":
                return []
            else:
                raise NeisAPIError(f"NEIS API error {code}: {message}")

        rows: List[Dict[str, Any]] = []
        for section in dataset:
            section_rows = section.get("row")
            if isinstance(section_rows, list):
                rows.extend(section_rows)
        return rows

    def search_schools(
        self,
        school_name: str,
        *,
        region_code: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        rows = self.request(
            "schoolInfo",
            SCHUL_NM=school_name,
            ATPT_OFCDC_SC_CODE=region_code,
            pIndex=page,
            pSize=page_size,
        )
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "education_office_code": row.get("ATPT_OFCDC_SC_CODE"),
                    "education_office_name": row.get("ATPT_OFCDC_SC_NM"),
                    "school_code": row.get("SD_SCHUL_CODE"),
                    "school_name": row.get("SCHUL_NM"),
                    "english_name": row.get("ENG_SCHUL_NM"),
                    "school_type": row.get("SCHUL_KND_SC_NM"),
                    "region_name": row.get("LCTN_SC_NM"),
                    "foundation": row.get("FOND_SC_NM"),
                    "coeducation": row.get("COEDU_SC_NM"),
                    "address": row.get("ORG_RDNMA"),
                    "address_detail": row.get("ORG_RDNDA") or row.get("ORG_RDNMA"),
                    "postal_code": row.get("ORG_RDNZIP")
                    or row.get("ORG_ZIPNO")
                    or row.get("ORG_ZIP_CODE"),
                    "telephone": row.get("ORG_TELNO"),
                    "fax": row.get("ORG_FAXNO"),
                    "homepage": row.get("HMPG_ADRES"),
                    "established_date": row.get("FOND_YMD"),
                }
            )
        return result

    def meal_service(
        self,
        *,
        region_code: str,
        school_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        meal_code: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        params = {
            "ATPT_OFCDC_SC_CODE": region_code,
            "SD_SCHUL_CODE": school_code,
            "MMEAL_SC_CODE": meal_code,
            "pIndex": page,
            "pSize": page_size,
        }
        if date:
            params["MLSV_YMD"] = date
        else:
            if start_date:
                params["MLSV_FROM_YMD"] = start_date
            if end_date:
                params["MLSV_TO_YMD"] = end_date

        rows = self.request("mealServiceDietInfo", **params)
        result: List[Dict[str, Any]] = []
        for row in rows:
            dishes = _parse_dishes(row.get("DDISH_NM"))
            result.append(
                {
                    "date": row.get("MLSV_YMD"),
                    "meal_code": row.get("MMEAL_SC_CODE"),
                    "meal_name": row.get("MMEAL_SC_NM"),
                    "calories": row.get("CAL_INFO"),
                    "dishes": dishes,
                    "origin_info": row.get("ORPLC_INFO"),
                    "nutrition_info": row.get("NTR_INFO"),
                    "school_name": row.get("SCHUL_NM"),
                }
            )
        return result

    def timetable(
        self,
        *,
        school_level: str,
        region_code: str,
        school_code: str,
        grade: str,
        class_name: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        endpoint = TIMETABLE_ENDPOINTS.get(school_level.lower())
        if not endpoint:
            raise NeisAPIError(
                f"Unsupported school_level '{school_level}'. Use one of: "
                f"{', '.join(sorted(TIMETABLE_ENDPOINTS))}."
            )

        params = {
            "ATPT_OFCDC_SC_CODE": region_code,
            "SD_SCHUL_CODE": school_code,
            "GRADE": grade,
            "CLASS_NM": class_name,
            "pIndex": page,
            "pSize": page_size,
        }
        if date:
            params["TI_YMD"] = date
        else:
            if start_date:
                params["TI_FROM_YMD"] = start_date
            if end_date:
                params["TI_TO_YMD"] = end_date

        rows = self.request(endpoint, **params)
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "date": row.get("ALL_TI_YMD"),
                    "period": row.get("PERIO"),
                    "subject_name": row.get("ITRT_CNTNT"),
                    "assembly_name": row.get("CLASS_NM"),
                    "grade": row.get("GRADE"),
                    "teacher": row.get("TEA_NM"),
                }
            )
        return result

    def academic_schedule(
        self,
        *,
        region_code: str,
        school_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        params = {
            "ATPT_OFCDC_SC_CODE": region_code,
            "SD_SCHUL_CODE": school_code,
            "pIndex": page,
            "pSize": page_size,
        }
        if date:
            params["AA_YMD"] = date
        else:
            if start_date:
                params["AA_FROM_YMD"] = start_date
            if end_date:
                params["AA_TO_YMD"] = end_date

        rows = self.request("SchoolSchedule", **params)
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "date": row.get("AA_YMD"),
                    "event_name": row.get("EVENT_NM"),
                    "event_content": row.get("EVENT_CNTNT"),
                    "grade": row.get("GRADE"),
                    "assembly_name": row.get("CLASS_NM"),
                    "event_type": row.get("EVENT_NM"),
                }
            )
        return result


def _parse_dishes(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    separators = ("<br/>", "<br>", "\\n", "\n")
    dishes: Iterable[str] = [raw_value]
    for sep in separators:
        dishes = _split_dishes(dishes, sep)
    return [dish.strip() for dish in dishes if dish.strip()]


def _split_dishes(values: Iterable[str], separator: str) -> List[str]:
    result: List[str] = []
    for value in values:
        result.extend(value.split(separator))
    return result


_neis_client: Optional[NeisClient] = None


def get_client() -> NeisClient:
    global _neis_client
    if _neis_client is None:
        _neis_client = NeisClient()
    return _neis_client


mcp = FastMCP("NEIS MCP")


@mcp.tool(description="교육청 코드 목록을 확인합니다.")
def list_education_office_codes() -> Dict[str, str]:
    return OFFICE_OF_EDUCATION_CODES


@mcp.tool(
    description="학교 이름과 (선택적으로) 교육청 코드를 사용해 NEIS에서 학교 기본정보를 검색합니다."
)
def search_schools(
    school_name: str,
    region_code: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    학교를 검색한 뒤 코드와 주소, 연락처 등의 기본정보를 반환합니다.
    """

    client = get_client()
    return client.search_schools(
        school_name,
        region_code=region_code,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    description="NEIS 급식 API를 통해 특정 학교의 급식 메뉴를 조회합니다. 날짜는 YYYYMMDD 형식입니다."
)
def get_school_meals(
    region_code: str,
    school_code: str,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    meal_code: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    급식 데이터를 조회합니다.
    - date: 특정 일자의 급식. 제공되면 기간(start/end)은 무시됩니다.
    - start_date/end_date: 기간을 지정할 때 사용합니다.
    - meal_code: 1(조식), 2(중식), 3(석식).
    """

    client = get_client()
    return client.meal_service(
        region_code=region_code,
        school_code=school_code,
        date=date,
        start_date=start_date,
        end_date=end_date,
        meal_code=meal_code,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    description="NEIS 시간표 API를 통해 특정 학년/학급의 시간표를 조회합니다. 날짜는 YYYYMMDD 형식입니다."
)
def get_school_timetable(
    school_level: str,
    region_code: str,
    school_code: str,
    grade: str,
    class_name: str,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    TimeTable 데이터를 조회합니다.
    school_level은 elementary, middle, high, special 중 하나입니다.
    """

    client = get_client()
    return client.timetable(
        school_level=school_level,
        region_code=region_code,
        school_code=school_code,
        grade=grade,
        class_name=class_name,
        date=date,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    description="학사일정 API를 통해 학교 행사 및 일정을 조회합니다. 날짜는 YYYYMMDD 형식입니다."
)
def get_academic_schedule(
    region_code: str,
    school_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    학사일정 데이터를 조회합니다. date가 지정되면 단일 일자, 그렇지 않으면 기간을 사용합니다.
    """

    client = get_client()
    return client.academic_schedule(
        region_code=region_code,
        school_code=school_code,
        start_date=start_date,
        end_date=end_date,
        date=date,
        page=page,
        page_size=page_size,
    )


@mcp.tool(description="MCP 서버 정보를 확인합니다.")
def get_server_info() -> Dict[str, Any]:
    return {
        "server_name": "NEIS MCP",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.sys.version.split()[0],
        "requires_api_key": True,
        "base_url": get_client().base_url,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    print(f"Starting FastMCP server on {host}:{port}")
    mcp.run(transport="http", host=host, port=port, stateless_http=True)
