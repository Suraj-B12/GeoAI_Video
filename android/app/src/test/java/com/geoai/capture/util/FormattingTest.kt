package com.geoai.capture.util

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

class FormattingTest {

    @Test fun numbersUseDotEvenInCommaLocales() {
        val saved = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY)
            assertEquals("12.971599000", Fmt.fixed(12.971599, 9))
            assertEquals("0.25", Fmt.opt(0.25f))
            assertEquals("", Fmt.opt(Double.NaN))
        } finally {
            Locale.setDefault(saved)
        }
    }

    @Test fun csvFieldsAreQuotedOnlyWhenNeeded() {
        assertEquals("plain", Fmt.field("plain"))
        assertEquals("\"a,b\"", Fmt.field("a,b"))
        assertEquals("\"say \"\"hi\"\"\"", Fmt.field("say \"hi\""))
        assertEquals("\"{\"\"k\"\":1,\"\"v\"\":2}\"", Fmt.field("{\"k\":1,\"v\":2}"))
    }

    @Test fun canonicalJsonSortsKeysWithoutWhitespace() {
        val o = JSONObject()
            .put("session_id", "S1")
            .put("files", JSONArray().put(JSONObject().put("sha256", "ab").put("path", "a/b.csv").put("bytes", 10L)))
            .put("created_wall_ms", 1700000000000L)
            .put("spec_version", "1.1")
        // Same bytes as Python json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assertEquals(
            "{\"created_wall_ms\":1700000000000,\"files\":[{\"bytes\":10,\"path\":\"a/b.csv\",\"sha256\":\"ab\"}],\"session_id\":\"S1\",\"spec_version\":\"1.1\"}",
            CanonicalJson.encode(o),
        )
    }

    @Test fun canonicalJsonEscapesLikePython() {
        assertEquals("\"a\\\"b\\\\c\\n\\u0001é\"", CanonicalJson.encode("a\"b\\c\n\u0001é"))
    }
}
