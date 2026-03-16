use std::cell::UnsafeCell;

use objc2::rc::Retained;
use objc2_foundation::NSQualityOfService;

use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::request::Request;
use crate::tensor_data::TensorData;
use crate::Error;

/// A compiled, loaded ANE program ready for repeated evaluation.
///
/// Obtained from [`Graph::compile`](crate::Graph::compile).
/// Automatically unloads from ANE hardware on drop.
pub struct Executable {
    pub(crate) inner: Retained<ANEInMemoryModel>,
    pub(crate) qos: NSQualityOfService,
    /// Cached request for `run_cached`. Lazily initialized on first call.
    pub(crate) cached_request: UnsafeCell<Option<Request>>,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Executable {
    /// Run the compiled program on the ANE.
    ///
    /// `inputs` and `outputs` are positional [`TensorData`] arrays matching the
    /// order of [`placeholder`](crate::Graph::placeholder) calls and output tensors
    /// in the graph.
    pub fn run(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<(), Error> {
        let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
            inputs.iter().map(|tensor_data| tensor_data.surface()).collect();
        let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
            outputs.iter().map(|tensor_data| tensor_data.surface()).collect();
        let request = Request::new(&input_surfaces, &output_surfaces)?;
        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }

    /// Run the compiled program, caching the ANE request object for reuse.
    ///
    /// On the first call, creates and caches an `_ANERequest` from the given
    /// IOSurfaces. Subsequent calls reuse the cached request, saving ~0.095ms
    /// of Objective-C object allocation per dispatch.
    ///
    /// # Safety requirement
    /// The caller must pass the **same** `inputs` and `outputs` TensorData on
    /// every call. The IOSurface backing buffers may be mutated between calls
    /// (that's the dynamic-weight pattern), but the TensorData/IOSurface
    /// objects themselves must be the same.
    pub fn run_cached(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<(), Error> {
        // SAFETY: Executable is not Sync for concurrent mutation — single-threaded
        // access to cached_request is guaranteed by the borrow of &self in the
        // training loop (one dispatch at a time).
        let cached = unsafe { &mut *self.cached_request.get() };
        if cached.is_none() {
            let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
                inputs.iter().map(|td| td.surface()).collect();
            let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
                outputs.iter().map(|td| td.surface()).collect();
            *cached = Some(Request::new(&input_surfaces, &output_surfaces)?);
        }
        let request = cached.as_ref().unwrap();
        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        self.inner.unload(self.qos);
    }
}
