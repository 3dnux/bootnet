package dev.bootnet.shieldtube

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Crea el motor (o lo recupera de la caché) y prepara los scripts cosméticos. */
object Shields {
    private const val TAG = "ShieldTube"

    @Volatile
    var engine: AdblockEngine? = null
        private set

    /** Bloqueante: llamar fuera del hilo principal. */
    fun load(context: Context): AdblockEngine? {
        engine?.let { return it }
        val serialized = File(context.filesDir, "engine.dat")
        val fresh = serialized.exists() &&
            System.currentTimeMillis() - serialized.lastModified() < FilterLists.MAX_AGE_MS
        var loaded = if (fresh) AdblockEngine.deserialize(serialized.readBytes()) else null
        if (loaded == null) {
            val lists = FilterLists.load(File(context.cacheDir, "filter-lists"))
            if (lists.isNotEmpty()) {
                loaded = AdblockEngine.build(lists)
                runCatching { serialized.writeBytes(loaded.serialize()) }
                Log.i(TAG, "Motor creado con ${lists.size} listas de Brave")
            } else if (serialized.exists()) {
                // Sin red: mejor un motor viejo que ninguno.
                loaded = AdblockEngine.deserialize(serialized.readBytes())
            }
        }
        loaded?.useResources(context.assets.open("resources.json").bufferedReader().use { it.readText() })
        engine = loaded
        return loaded
    }

    /** Excepciones de filtros genéricos de un origen, para hiddenClassIdSelectors. */
    fun exceptionsFor(engine: AdblockEngine, origin: String): String =
        JSONObject(engine.cosmeticResources("$origin/")).optJSONArray("exceptions")?.toString() ?: "[]"

    /**
     * Script que se ejecuta al empezar cada documento de `origin`, antes que los de la
     * página: scriptlets de Brave/uBO + arranque de shields.js con el CSS y los
     * filtros procedurales de ese sitio.
     */
    fun documentStartScript(context: Context, engine: AdblockEngine, origin: String): String {
        val res = JSONObject(engine.cosmeticResources("$origin/"))
        val selectors = res.optJSONArray("hide_selectors") ?: JSONArray()
        val css = (0 until selectors.length()).joinToString("\n") { "${selectors.getString(it)}{display:none!important}" }
        val config = JSONObject()
            .put("css", css)
            .put("generichide", res.optBoolean("generichide"))
            .put("procedural", res.optJSONArray("procedural_actions") ?: JSONArray())
        val bootstrap = context.assets.open("shields.js").bufferedReader().use { it.readText() }
        return "${res.optString("injected_script")}\n;try{($bootstrap)($config);}catch(e){}"
    }
}
