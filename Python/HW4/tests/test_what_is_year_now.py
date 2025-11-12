import unittest
from unittest.mock import patch, MagicMock
import io
import json
from what_is_year_now import what_is_year_now

class TestWhatIsYearNow(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_ymd_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_urlopen.return_value = mock_resp
        mock_data = {"currentDateTime": "2019-03-01"}
        with patch("json.load", return_value=mock_data):
            assert what_is_year_now() == 2019

    @patch("urllib.request.urlopen")
    def test_dmy_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_urlopen.return_value = mock_resp
        mock_data = {"currentDateTime": "01.03.2019"}
        with patch("json.load", return_value=mock_data):
            assert what_is_year_now() == 2019

    @patch("urllib.request.urlopen")
    def test_invalid_format_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_urlopen.return_value = mock_resp
        mock_data = {"currentDateTime": "20190301"}
        with patch("json.load", return_value=mock_data):
            with self.assertRaises(ValueError):
                what_is_year_now()

if __name__ == '__main__':
    unittest.main()
