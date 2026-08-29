/**
 * Projects exported for the Web expose the :js:class:`Engine` class to the JavaScript environment, that allows
 * fine control over the engine's start-up process.
 *
 * This API is built in an asynchronous manner and requires basic understanding
 * of `Promises <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises>`__.
 *
 * @module Engine
 * @header Web export JavaScript reference
 */
const Engine = (function () {
	const preloader = new Preloader();

	let loadPromise = null;
	let loadPath = '';
	let initPromise = null;

	/**
	 * @classdesc The ``Engine`` class provides methods for loading and starting exported projects on the Web. For default export
	 * settings, this is already part of the exported HTML page. To understand practical use of the ``Engine`` class,
	 * see :ref:`Custom HTML page for Web export <doc_customizing_html5_shell>`.
	 *
	 * @description Create a new Engine instance with the given configuration.
	 *
	 * @global
	 * @constructor
	 * @param {EngineConfig} initConfig The initial config for this instance.
	 */
	function Engine(initConfig) { // eslint-disable-line no-shadow
		this.config = new InternalConfig(initConfig);
		this.rtenv = null;
	}

	/**
	 * Load the engine from the specified base path.
	 *
	 * @param {string} basePath Base path of the engine to load.
	 * @param {number=} [size=0] The file size if known.
	 * @returns {Promise} A Promise that resolves once the engine is loaded.
	 *
	 * @function Engine.load
	 */
	Engine.load = function (basePath, size) {
		if (loadPromise == null) {
			loadPath = basePath;
			loadPromise = preloader.loadPromise(`${loadPath}.wasm`, size, true);
			requestAnimationFrame(preloader.animateProgress);
		}
		return loadPromise;
	};

	/**
	 * Unload the engine to free memory.
	 *
	 * This method will be called automatically depending on the configuration. See :js:attr:`unloadAfterInit`.
	 *
	 * @function Engine.unload
	 */
	Engine.unload = function () {
		loadPromise = null;
	};

	/**
	 * Safe Engine constructor, creates a new prototype for every new instance to avoid prototype pollution.
	 * @ignore
	 * @constructor
	 */
	function SafeEngine(initConfig) {
		const proto = /** @lends Engine.prototype */ {
			/**
			 * Initialize the engine instance. Optionally, pass the base path to the engine to load it,
			 * if it hasn't been loaded yet. See :js:meth:`Engine.load`.
			 *
			 * @param {string=} basePath Base path of the engine to load.
			 * @return {Promise} A ``Promise`` that resolves once the engine is loaded and initialized.
			 */
			init: function (basePath) {
				if (initPromise) {
					return initPromise;
				}
				if (loadPromise == null) {
					if (!basePath) {
						initPromise = Promise.reject(new Error('A base path must be provided when calling `init` and the engine is not loaded.'));
						return initPromise;
					}
					Engine.load(basePath, this.config.fileSizes[`${basePath}.wasm`]);
				}
				const me = this;
				function doInit(promise) {
					// Care! Promise chaining is bogus with old emscripten versions.
					// This caused a regression with the Mono build (which uses an older emscripten version).
					// Make sure to test that when refactoring.
					return new Promise(function (resolve, reject) {
						promise.then(function (response) {
							const cloned = new Response(response.clone().body, { 'headers': [['content-type', 'application/wasm']] });
							Godot(me.config.getModuleConfig(loadPath, cloned)).then(function (module) {
								const paths = me.config.persistentPaths;
								module['initFS'](paths).then(function (err) {
									me.rtenv = module;
									if (me.config.unloadAfterInit) {
										Engine.unload();
									}
									resolve();
								});
							});
						});
					});
				}
				preloader.setProgressFunc(this.config.onProgress);
				initPromise = doInit(loadPromise);
				return initPromise;
			},

			/**
			 * Load a file so it is available in the instance's file system once it runs. Must be called **before** starting the
			 * instance.
			 *
			 * If not provided, the ``path`` is derived from the URL of the loaded file.
			 *
			 * @param {string|ArrayBuffer} file The file to preload.
			 *
			 * If a ``string`` the file will be loaded from that path.
			 *
			 * If an ``ArrayBuffer`` or a view on one, the buffer will used as the content of the file.
			 *
			 * @param {string=} path Path by which the file will be accessible. Required, if ``file`` is not a string.
			 *
			 * @returns {Promise} A Promise that resolves once the file is loaded.
			 */
			preloadFile: function (file, path) {
				return preloader.preload(file, path, this.config.fileSizes[file]);
			},

			/**
			 * Start the engine instance using the given override configuration (if any).
			 * :js:meth:`startGame <Engine.prototype.startGame>` can be used in typical cases instead.
			 *
			 * This will initialize the instance if it is not initialized. For manual initialization, see :js:meth:`init <Engine.prototype.init>`.
			 * The engine must be loaded beforehand.
			 *
			 * Fails if a canvas cannot be found on the page, or not specified in the configuration.
			 *
			 * @param {EngineConfig} override An optional configuration override.
			 * @return {Promise} Promise that resolves once the engine started.
			 */
			start: function (override) {
				this.config.update(override);
				const me = this;
				return me.init().then(function () {
					if (!me.rtenv) {
						return Promise.reject(new Error('The engine must be initialized before it can be started'));
					}

					let config = {};
					try {
						config = me.config.getGodotConfig(function () {
							me.rtenv = null;
						});
					} catch (e) {
						return Promise.reject(e);
					}
					// Godot configuration.
					me.rtenv['initConfig'](config);

					// Preload GDExtension libraries.
					if (me.config.gdextensionLibs.length > 0 && !me.rtenv['loadDynamicLibrary']) {
						return Promise.reject(new Error('GDExtension libraries are not supported by this engine version. '
							+ 'Enable "Extensions Support" for your export preset and/or build your custom template with "dlink_enabled=yes".'));
					}
					return new Promise(function (resolve, reject) {
						for (const file of preloader.preloadedFiles) {
							me.rtenv['copyToFS'](file.path, file.buffer);
						}
						preloader.preloadedFiles.length = 0; // Clear memory
						// Once the download completes the progress bar sits at 100%
						// while `callMain` runs the engine's synchronous start-up
						// (module/physics init and, on WebGPU, shader/pipeline
						// compilation). That work blocks the main thread with no
						// progress reporting, so the page can look frozen. Surface
						// a boot-phase message and force it to paint *before* the
						// blocking call, so the last thing drawn is legible text
						// instead of a stalled bar.
						function runMain() {
							me.rtenv['callMain'](me.config.args);
							initPromise = null;
							me.installServiceWorker();
							resolve();
						}
						if (typeof me.config.onStatusChange === 'function') {
							// Machine-readable phase token; the shell owns the (localized)
							// display text. 'shader-compile' only when a WebGPU device was
							// actually pre-initialized — with renderingDriver=webgpu but no
							// device, the engine falls back to GL and 'engine-init' is the
							// honest phase.
							const phase = (me.config.renderingDriver === 'webgpu' && me.config.preinitializedWebGPUDevice)
								? 'shader-compile'
								: 'engine-init';
							me.config.onStatusChange(phase);
							// Yield to the browser so the message actually paints before
							// `callMain` blocks the thread. A double rAF is NOT a
							// composite guarantee: measured under post-download load
							// (CDP screencast), the two rAF callbacks fired 2-8ms apart
							// with no frame between them, and the message never reached
							// a single compositor frame. Instead, align to a frame
							// boundary with one rAF, then give the compositor real time
							// to present the committed change before blocking. rAF is
							// suspended entirely in background tabs, so keep an absolute
							// fallback timer (run-once latch) or a backgrounded page
							// would not boot until focused.
							let mainStarted = false;
							const runMainOnce = function () {
								if (!mainStarted) {
									mainStarted = true;
									runMain();
								}
							};
							if (typeof requestAnimationFrame === 'function') {
								requestAnimationFrame(function () {
									setTimeout(runMainOnce, 100);
								});
							}
							setTimeout(runMainOnce, 1000);
						} else {
							runMain();
						}
					});
				});
			},

			/**
			 * Start the game instance using the given configuration override (if any).
			 *
			 * This will initialize the instance if it is not initialized. For manual initialization, see :js:meth:`init <Engine.prototype.init>`.
			 *
			 * This will load the engine if it is not loaded, and preload the main pck.
			 *
			 * This method expects the initial config (or the override) to have both the :js:attr:`executable` and :js:attr:`mainPack`
			 * properties set (normally done by the editor during export).
			 *
			 * @param {EngineConfig} override An optional configuration override.
			 * @return {Promise} Promise that resolves once the game started.
			 */
			startGame: function (override) {
				this.config.update(override);
				// Add main-pack argument.
				const exe = this.config.executable;
				const pack = this.config.mainPack || `${exe}.pck`;
				this.config.args = ['--main-pack', pack].concat(this.config.args);
				// Start and init with execName as loadPath if not inited.
				const me = this;

				// If WebGPU rendering is requested, pre-initialize a GPUDevice
				// before the WASM module starts (the C++ side expects it in
				// Module.preinitializedWebGPUDevice).
				// SPIR-V→WGSL conversion is handled by Tint, compiled directly
				// into the engine WASM — no separate translator module needed.
				let webgpuReady = Promise.resolve();
				if (me.config.renderingDriver === 'webgpu' && !me.config.preinitializedWebGPUDevice) {
					webgpuReady = Engine.requestWebGPUDevice().then(function (device) {
						me.config.preinitializedWebGPUDevice = device;
					});
				}

				return webgpuReady.then(function () {
					return Promise.all([
						me.init(exe),
						me.preloadFile(pack, pack),
					]).then(function () {
						return me.start.apply(me);
					});
				});
			},

			/**
			 * Create a file at the specified ``path`` with the passed as ``buffer`` in the instance's file system.
			 *
			 * @param {string} path The location where the file will be created.
			 * @param {ArrayBuffer} buffer The content of the file.
			 */
			copyToFS: function (path, buffer) {
				if (this.rtenv == null) {
					throw new Error('Engine must be inited before copying files');
				}
				this.rtenv['copyToFS'](path, buffer);
			},

			/**
			 * Request that the current instance quit.
			 *
			 * This is akin the user pressing the close button in the window manager, and will
			 * have no effect if the engine has crashed, or is stuck in a loop.
			 *
			 */
			requestQuit: function () {
				if (this.rtenv) {
					this.rtenv['request_quit']();
				}
			},

			/**
			 * Install the progressive-web app service worker.
			 * @returns {Promise} The service worker registration promise.
			 */
			installServiceWorker: function () {
				if (this.config.serviceWorker && 'serviceWorker' in navigator) {
					try {
						return navigator.serviceWorker.register(this.config.serviceWorker);
					} catch (e) {
						return Promise.reject(e);
					}
				}
				return Promise.resolve();
			},
		};

		Engine.prototype = proto;
		// Closure compiler exported instance methods.
		Engine.prototype['init'] = Engine.prototype.init;
		Engine.prototype['preloadFile'] = Engine.prototype.preloadFile;
		Engine.prototype['start'] = Engine.prototype.start;
		Engine.prototype['startGame'] = Engine.prototype.startGame;
		Engine.prototype['copyToFS'] = Engine.prototype.copyToFS;
		Engine.prototype['requestQuit'] = Engine.prototype.requestQuit;
		Engine.prototype['installServiceWorker'] = Engine.prototype.installServiceWorker;
		// Also expose static methods as instance methods
		Engine.prototype['load'] = Engine.load;
		Engine.prototype['unload'] = Engine.unload;
		return new Engine(initConfig);
	}

	/**
	 * Request a WebGPU device from the browser. Returns a Promise that resolves
	 * to a ``GPUDevice``, or rejects if WebGPU is not available.
	 *
	 * This is called automatically by :js:meth:`startGame <Engine.prototype.startGame>`
	 * when ``renderingDriver`` is ``'webgpu'``.
	 *
	 * @param {Object=} adapterOptions Optional ``GPURequestAdapterOptions``.
	 * @param {Object=} deviceDescriptor Optional ``GPUDeviceDescriptor``.
	 * @returns {Promise<GPUDevice>} A promise resolving to the GPU device.
	 *
	 * @function Engine.requestWebGPUDevice
	 */
	Engine.requestWebGPUDevice = function (adapterOptions, deviceDescriptor) {
		if (typeof navigator === 'undefined' || !navigator.gpu) {
			return Promise.reject(new Error(
				'WebGPU is not supported in this browser.\n'
				+ 'Try using a recent version of Chrome, Edge, or Firefox.'
			));
		}
		return navigator.gpu.requestAdapter(adapterOptions || {
			powerPreference: 'high-performance',
		}).then(function (adapter) {
			if (!adapter) {
				return Promise.reject(new Error(
					'WebGPU adapter not found. Your GPU may not support WebGPU.'
				));
			}
			const desc = deviceDescriptor || {
				requiredFeatures: [],
			};
			// Request timestamp-query if available.
			if (adapter.features.has('timestamp-query')) {
				desc.requiredFeatures = (desc.requiredFeatures || []).concat(['timestamp-query']);
			}
			// Request optional texture format tiers used by Godot (r16snorm, rg16snorm, etc.).
			var optionalFeatures = [
				'readonly-and-readwrite-storage-textures',
				'texture-formats-tier1',
				'texture-formats-tier2',
				'float32-filterable',
				'float32-blendable',
				'rg11b10ufloat-renderable',
				'clip-distances',
				'dual-source-blending',
				// Texture compression families — without these, Godot's DXT5/ETC2/ASTC
				// assets get decompressed on the CPU to RGBA8, which triggers a warning
				// and wastes VRAM. The driver only reports the formats as supported
				// when the corresponding feature is actually enabled here.
				'texture-compression-bc',
				'texture-compression-etc2',
				'texture-compression-astc',
			];
			for (var i = 0; i < optionalFeatures.length; i++) {
				if (adapter.features.has(optionalFeatures[i])) {
					desc.requiredFeatures = (desc.requiredFeatures || []).concat([optionalFeatures[i]]);
				}
			}
			// Request higher limits that Godot's renderer needs.
			// The adapter may support more than the default; request what it offers.
			var adapterLimits = adapter.limits || {};
			desc.requiredLimits = desc.requiredLimits || {};
			var limitsToMax = [
				'maxStorageBuffersPerShaderStage',
				'maxStorageBufferBindingSize',
				'maxBufferSize',
				'maxUniformBufferBindingSize',
				'maxUniformBuffersPerShaderStage',
				'maxSampledTexturesPerShaderStage',
				'maxSamplersPerShaderStage',
				'maxColorAttachments',
				'maxBindGroups',
			];
			for (var li = 0; li < limitsToMax.length; li++) {
				var key = limitsToMax[li];
				if (adapterLimits[key] !== undefined) {
					desc.requiredLimits[key] = adapterLimits[key];
				}
			}
			return adapter.requestDevice(desc).then(function (device) {
				// Monitor device loss (non-blocking — just log).
				device.lost.then(function (info) {
					const reason = info.reason || 'unknown';
					const msg = info.message || '';
					console.error(`[Godot] WebGPU device lost (reason: ${reason}): ${msg}`);
				});
				device.addEventListener('uncapturederror', function (event) {
					// Include the full error message — `event.error` alone only
					// serializes to the constructor name ("GPUValidationError"),
					// hiding the detail. `.message` contains the Dawn error text.
					const err = event.error;
					const kind = (err && err.constructor && err.constructor.name) || 'UnknownError';
					const msg = (err && err.message) ? err.message : String(err);
					console.error(`[Godot] WebGPU uncaptured error: ${kind}: ${msg}`);
				});
				return device;
			});
		});
	};

	// Closure compiler exported static methods.
	SafeEngine['load'] = Engine.load;
	SafeEngine['unload'] = Engine.unload;
	SafeEngine['requestWebGPUDevice'] = Engine.requestWebGPUDevice;

	// Feature-detection utilities.
	SafeEngine['isWebGLAvailable'] = Features.isWebGLAvailable;
	SafeEngine['isFetchAvailable'] = Features.isFetchAvailable;
	SafeEngine['isSecureContext'] = Features.isSecureContext;
	SafeEngine['isCrossOriginIsolated'] = Features.isCrossOriginIsolated;
	SafeEngine['isSharedArrayBufferAvailable'] = Features.isSharedArrayBufferAvailable;
	SafeEngine['isAudioWorkletAvailable'] = Features.isAudioWorkletAvailable;
	SafeEngine['getMissingFeatures'] = Features.getMissingFeatures;

	return SafeEngine;
}());
if (typeof window !== 'undefined') {
	window['Engine'] = Engine;
}
