# Frontend

The frontend is a small React application written in TypeScript and built with Vite. It currently provides an image upload screen with client-side validation and a local preview. It does not yet send images to the backend or run a deepfake detection model.

## How the code runs

`index.html` provides the page shell and the `#root` element. Vite serves this page during development and bundles it for production. The entry point, [`src/main.tsx`](src/main.tsx), imports the global stylesheet and mounts the React application inside `StrictMode`. It renders [`src/App.tsx`](src/App.tsx), which contains the current upload experience.

The `App` component keeps three pieces of state:

- `selectedFile`: the accepted `File` object. It is saved for future use, but currently is not submitted anywhere.
- `previewUrl`: a temporary browser object URL used to display the uploaded image.
- `error`: a validation message shown to the user with `role="alert"`.

When the file input changes, `handleFileChange` checks the first selected file. Only JPEG, PNG, and WebP images up to 5 MiB are accepted. On success, the component clears any old error, saves the file, and creates an object URL for the preview. On failure, `handleError` clears both the selected file and preview and sets an explanatory message. A `useEffect` cleanup revokes the current object URL when it changes or the component unmounts, releasing the browser-held resource.

The page displays a heading, brief instructions, a native file picker, and—when valid input is selected—an `<img>` preview. The `accept` attribute guides the picker, while the handler performs the actual checks.

## Source layout

- `src/main.tsx` — React mount point and global CSS import.
- `src/App.tsx` — upload UI, state, validation, and preview lifecycle.
- `src/index.css` — global typography, color variables, page layout, and light/dark color scheme.
- `src/App.css` — component-level styles. Some rules are leftover starter-template styles and are not used by the current upload screen.
- `public/` — static assets served from the site root.
- `vite.config.ts` — enables Vite's React plugin.
- `tsconfig*.json` — TypeScript compiler settings for the app and build tooling.

## Development commands

Run these from the `frontend` directory:

```sh
npm install
npm run dev
```

Vite prints the local development URL in the terminal and supports hot module replacement. Other available scripts are:

```sh
npm run build    # type-check and create the production bundle in dist/
npm run preview  # serve the production bundle locally
npm run lint     # run ESLint
```

## Current scope and next integration point

This is currently a browser-only upload and preview flow. There is no API client, request state, result view, or backend URL configuration in the frontend yet. The accepted `selectedFile` is the natural input for a future upload request; that integration can add request progress, API error handling, and a detection result to `App` or move those responsibilities into dedicated components/hooks as the UI grows.
