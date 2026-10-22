//! Puente JNI entre la app Android y adblock-rust, el motor de Brave Shields.
//!
//! Los punteros (`jlong`) que se devuelven a Kotlin son `Box`es; Kotlin es
//! responsable de liberarlos con `nativeFree`/`nativeBuildEngine`.

use std::collections::HashSet;
use std::sync::RwLock;

use adblock::lists::{FilterFormat, FilterSet, ParseOptions};
use adblock::request::Request;
use adblock::resources::{PermissionMask, Resource};
use adblock::Engine;
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jboolean, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

pub type SharedEngine = RwLock<Engine>;

/// Resultado de una consulta de red, en el formato que entiende Kotlin:
/// `None` = permitir, `"B"` = bloquear, `"R<url>"` = responder con el recurso `url`.
pub fn check(engine: &Engine, url: &str, source_url: &str, request_type: &str, method: &str) -> Option<String> {
    let request = Request::new(url, source_url, request_type, method).ok()?;
    let result = engine.check_network_request(&request);
    if !result.should_block() {
        return None;
    }
    match result.redirect {
        Some(redirect) => Some(format!("R{redirect}")),
        None => Some("B".to_string()),
    }
}

/// Recursos cosméticos (CSS, scriptlets, procedurales) para `url`, en JSON.
pub fn cosmetic_json(engine: &Engine, url: &str) -> String {
    serde_json::to_string(&engine.url_cosmetic_resources(url)).unwrap_or_default()
}

/// Selectores genéricos a ocultar para las clases/ids dados (arrays JSON).
pub fn class_id_selectors_json(engine: &Engine, classes: &str, ids: &str, exceptions: &str) -> String {
    let parse = |s: &str| serde_json::from_str::<Vec<String>>(s).unwrap_or_default();
    let exceptions: HashSet<String> = parse(exceptions).into_iter().collect();
    let selectors = engine.hidden_class_id_selectors(parse(classes), parse(ids), &exceptions);
    serde_json::to_string(&selectors).unwrap_or_default()
}

pub fn add_filters(set: &mut FilterSet, text: &str, permission: u8, hosts: bool) {
    let opts = ParseOptions {
        format: if hosts { FilterFormat::Hosts } else { FilterFormat::Standard },
        permissions: PermissionMask::from_bits(permission),
        ..ParseOptions::default()
    };
    set.add_filter_list(text.to_string(), opts);
}

pub fn use_resources_json(engine: &mut Engine, json: &str) -> bool {
    match serde_json::from_str::<Vec<Resource>>(json) {
        Ok(resources) => {
            engine.use_resources(resources);
            true
        }
        Err(_) => false,
    }
}

// --- JNI ---------------------------------------------------------------------

fn get_string(env: &mut JNIEnv, s: &JString) -> String {
    env.get_string(s).map(Into::into).unwrap_or_default()
}

fn new_string(env: &mut JNIEnv, s: &str) -> jstring {
    env.new_string(s).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

unsafe fn engine_ref<'a>(handle: jlong) -> &'a SharedEngine {
    &*(handle as *const SharedEngine)
}

fn into_handle(engine: Engine) -> jlong {
    Box::into_raw(Box::new(RwLock::new(engine))) as jlong
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeNewFilterSet(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    Box::into_raw(Box::new(FilterSet::new(false))) as jlong
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeAddFilters(
    mut env: JNIEnv,
    _class: JClass,
    set: jlong,
    text: JString,
    permission: jint,
    hosts: jboolean,
) {
    let text = get_string(&mut env, &text);
    let set = unsafe { &mut *(set as *mut FilterSet) };
    add_filters(set, &text, permission as u8, hosts == JNI_TRUE);
}

/// Consume el FilterSet y devuelve el motor.
#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeBuildEngine(
    _env: JNIEnv,
    _class: JClass,
    set: jlong,
) -> jlong {
    let set = unsafe { Box::from_raw(set as *mut FilterSet) };
    into_handle(Engine::new_with_filter_set(*set))
}

/// Devuelve 0 si los datos no son un motor serializado válido.
#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeDeserialize(
    env: JNIEnv,
    _class: JClass,
    data: JByteArray,
) -> jlong {
    let Ok(bytes) = env.convert_byte_array(&data) else { return 0 };
    let mut engine = Engine::default();
    match engine.deserialize(&bytes) {
        Ok(()) => into_handle(engine),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeSerialize<'a>(
    env: JNIEnv<'a>,
    _class: JClass,
    handle: jlong,
) -> JByteArray<'a> {
    let bytes = unsafe { engine_ref(handle) }.read().map(|e| e.serialize()).unwrap_or_default();
    env.byte_array_from_slice(&bytes).unwrap_or_default()
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeUseResources(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    json: JString,
) -> jboolean {
    let json = get_string(&mut env, &json);
    let ok = unsafe { engine_ref(handle) }
        .write()
        .map(|mut e| use_resources_json(&mut e, &json))
        .unwrap_or(false);
    if ok { JNI_TRUE } else { JNI_FALSE }
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeCheck(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    url: JString,
    source_url: JString,
    request_type: JString,
    method: JString,
) -> jstring {
    let url = get_string(&mut env, &url);
    let source_url = get_string(&mut env, &source_url);
    let request_type = get_string(&mut env, &request_type);
    let method = get_string(&mut env, &method);
    let result = unsafe { engine_ref(handle) }
        .read()
        .ok()
        .and_then(|e| check(&e, &url, &source_url, &request_type, &method));
    match result {
        Some(r) => new_string(&mut env, &r),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeCosmetic(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    url: JString,
) -> jstring {
    let url = get_string(&mut env, &url);
    let json = unsafe { engine_ref(handle) }.read().map(|e| cosmetic_json(&e, &url)).unwrap_or_default();
    new_string(&mut env, &json)
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeClassIdSelectors(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    classes: JString,
    ids: JString,
    exceptions: JString,
) -> jstring {
    let classes = get_string(&mut env, &classes);
    let ids = get_string(&mut env, &ids);
    let exceptions = get_string(&mut env, &exceptions);
    let json = unsafe { engine_ref(handle) }
        .read()
        .map(|e| class_id_selectors_json(&e, &classes, &ids, &exceptions))
        .unwrap_or_default();
    new_string(&mut env, &json)
}

#[no_mangle]
pub extern "system" fn Java_dev_bootnet_shieldtube_AdblockEngine_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        drop(unsafe { Box::from_raw(handle as *mut SharedEngine) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        let mut set = FilterSet::new(false);
        add_filters(
            &mut set,
            "||ads.example^\n||tracker.example/lib.js$script,redirect=noopjs\nvideo.example##.banner-ad\nvideo.example##+js(noop-scriptlet)",
            1,
            false,
        );
        add_filters(&mut set, "0.0.0.0 hosts.example", 0, true);
        let mut engine = Engine::new_with_filter_set(set);
        assert!(use_resources_json(
            &mut engine,
            r#"[{"name":"noopjs","aliases":[],"kind":{"mime":"application/javascript"},"content":"KGZ1bmN0aW9uKCl7fSkoKQ=="},
                {"name":"noop-scriptlet.js","aliases":[],"kind":"template","content":"d2luZG93Ll9fc2hpZWxkdHViZSA9IHRydWU7"}]"#,
        ));
        engine
    }

    #[test]
    fn blocks_and_redirects() {
        let e = engine();
        assert_eq!(check(&e, "https://ads.example/x.png", "https://video.example/", "image", "GET").as_deref(), Some("B"));
        assert_eq!(check(&e, "https://hosts.example/x", "https://video.example/", "xmlhttprequest", "GET").as_deref(), Some("B"));
        assert_eq!(check(&e, "https://video.example/app.js", "https://video.example/", "script", "GET"), None);
        let redirect = check(&e, "https://tracker.example/lib.js", "https://video.example/", "script", "GET").unwrap();
        assert!(redirect.starts_with("Rdata:application/javascript;base64,"));
    }

    #[test]
    fn cosmetics() {
        let e = engine();
        let json: serde_json::Value = serde_json::from_str(&cosmetic_json(&e, "https://video.example/watch")).unwrap();
        assert!(json["hide_selectors"].as_array().unwrap().iter().any(|s| s == ".banner-ad"));
        assert!(json["injected_script"].as_str().unwrap().contains("window.__shieldtube = true"));
    }

    #[test]
    fn serialization_roundtrip() {
        let e = engine();
        let mut restored = Engine::default();
        restored.deserialize(&e.serialize()).unwrap();
        assert_eq!(check(&restored, "https://ads.example/x.png", "https://video.example/", "image", "GET").as_deref(), Some("B"));
    }
}
