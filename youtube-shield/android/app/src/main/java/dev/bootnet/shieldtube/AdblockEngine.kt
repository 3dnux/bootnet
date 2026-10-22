package dev.bootnet.shieldtube

/**
 * Envoltorio del motor adblock-rust de Brave (libshieldtube.so, ver android/rust).
 * Es seguro usarlo desde varios hilos.
 */
class AdblockEngine private constructor(private val handle: Long) {

    /** Lista de filtros con el bit de permisos que Brave le asigna en su catálogo. */
    class FilterList(val text: String, val permission: Int, val hosts: Boolean)

    sealed interface Decision {
        data object Block : Decision
        data class Redirect(val dataUrl: String) : Decision
    }

    /** `null` = dejar pasar la petición. */
    fun check(url: String, sourceUrl: String, requestType: String, method: String): Decision? {
        val result = nativeCheck(handle, url, sourceUrl, requestType, method) ?: return null
        return if (result.startsWith("R")) Decision.Redirect(result.substring(1)) else Decision.Block
    }

    /** JSON con hide_selectors, injected_script, procedural_actions, exceptions y generichide. */
    fun cosmeticResources(url: String): String = nativeCosmetic(handle, url)

    /** Array JSON de selectores genéricos a ocultar. Argumentos: arrays JSON de strings. */
    fun hiddenClassIdSelectors(classes: String, ids: String, exceptions: String): String =
        nativeClassIdSelectors(handle, classes, ids, exceptions)

    fun useResources(json: String): Boolean = nativeUseResources(handle, json)

    fun serialize(): ByteArray = nativeSerialize(handle)

    companion object {
        init {
            System.loadLibrary("shieldtube")
        }

        fun build(lists: List<FilterList>): AdblockEngine {
            val set = nativeNewFilterSet()
            for (list in lists) nativeAddFilters(set, list.text, list.permission, list.hosts)
            return AdblockEngine(nativeBuildEngine(set))
        }

        fun deserialize(data: ByteArray): AdblockEngine? =
            nativeDeserialize(data).takeIf { it != 0L }?.let(::AdblockEngine)

        @JvmStatic private external fun nativeNewFilterSet(): Long
        @JvmStatic private external fun nativeAddFilters(set: Long, text: String, permission: Int, hosts: Boolean)
        @JvmStatic private external fun nativeBuildEngine(set: Long): Long
        @JvmStatic private external fun nativeDeserialize(data: ByteArray): Long
        @JvmStatic private external fun nativeSerialize(handle: Long): ByteArray
        @JvmStatic private external fun nativeUseResources(handle: Long, json: String): Boolean
        @JvmStatic private external fun nativeCheck(
            handle: Long, url: String, sourceUrl: String, requestType: String, method: String,
        ): String?
        @JvmStatic private external fun nativeCosmetic(handle: Long, url: String): String
        @JvmStatic private external fun nativeClassIdSelectors(
            handle: Long, classes: String, ids: String, exceptions: String,
        ): String
    }
}
