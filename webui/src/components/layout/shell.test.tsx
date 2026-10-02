import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  Nav,
  NavItem,
  NavSubItem,
  PageHeader,
  PageTransition,
  RowLink,
  Shell,
  Sidebar,
  SidebarBrand,
  Table,
} from "@/components/layout/shell";
import { Badge, Chip, IconButton } from "@/components/ui/index";
import { TooltipProvider } from "@/components/ui/tooltip";
import brandIcon from "../../../../assets/icon.svg";

describe("Shell", () => {
  it("places the sidebar beside the page without a notice slot", () => {
    const html = renderToStaticMarkup(
      <Shell sidebar={<nav>Navigation</nav>}>
        <main>Draft editor</main>
      </Shell>,
    );
    expect(html).toContain("<nav>Navigation</nav>");
    expect(html).toContain("<main>Draft editor</main>");
    expect(html).not.toContain("banner");
    expect(html).not.toContain('role="status"');
  });

  it("removes sidebar navigation from interaction while settings save", () => {
    const html = renderToStaticMarkup(
      <Sidebar inert>
        <Nav label="Sections">
          <NavItem
            icon={<span>i</span>}
            label="Members"
            active={false}
            onSelect={() => {}}
          />
        </Nav>
      </Sidebar>,
    );
    expect(html).toMatch(/<div[^>]*inert=""/);
  });
});

describe("SidebarBrand", () => {
  it("shows the official icon and brand name without a summary or interaction", () => {
    const html = renderToStaticMarkup(<SidebarBrand />);
    const icon = renderToStaticMarkup(<img src={brandIcon} alt="" />);
    expect(html.match(/<img\b[^>]*src="([^"]*)"/)?.[1]).toBe(
      icon.match(/<img\b[^>]*src="([^"]*)"/)?.[1],
    );
    expect(html).toMatch(/<img\b[^>]*alt=""/);
    expect(html).toMatch(/<span\b[^>]*>Fora<\/span>/);
    expect(html).not.toMatch(/Members?|running/);
    expect(html).not.toMatch(/<(?:button|a)\b|tabindex=|aria-label=/);
  });
});

describe("Sidebar navigation", () => {
  it("nests sub items under an item with a trailing action", () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <Sidebar footer={<span>Footer</span>}>
          <Nav label="Sections">
            <NavItem
              icon={<span>i</span>}
              label="Discussions"
              active={false}
              badge={<Badge tone="unread">3</Badge>}
              trailing={
                <IconButton size="sm" label="New Discussion">
                  +
                </IconButton>
              }
              onSelect={() => {}}
            >
              <NavSubItem
                label="Ship the release notes"
                active
                badge={<Badge tone="unread">1</Badge>}
                indicator={<i>dot</i>}
                onSelect={() => {}}
              />
              <NavSubItem
                label="Weekly review"
                active={false}
                onSelect={() => {}}
              />
            </NavItem>
            <NavItem
              icon={<span>i</span>}
              label="Members"
              active={false}
              onSelect={() => {}}
            />
          </Nav>
        </Sidebar>
      </TooltipProvider>,
    );
    expect(html).toContain('aria-label="New Discussion"');
    expect(html.match(/data-nav="subitem"/g)).toHaveLength(2);
    expect(html.match(/aria-current="page"/g)).toHaveLength(1);
    expect(html).toContain('aria-current="page" data-unread="true"');
    expect(html).toContain("<i>dot</i>");
    expect(html.match(/<ul\b/g)).toHaveLength(2);
    expect(html).toContain("<span>Footer</span>");
  });

  it("renders an item without children flat", () => {
    const html = renderToStaticMarkup(
      <Nav label="Sections">
        <NavItem
          icon={<span>i</span>}
          label="Library"
          active
          onSelect={() => {}}
        />
      </Nav>,
    );
    expect(html.match(/<ul\b/g)).toHaveLength(1);
    expect(html.match(/<button\b/g)).toHaveLength(1);
    expect(html).toContain('data-active="true"');
  });
});

describe("PageTransition", () => {
  it("wraps the page in the animated container", () => {
    const html = renderToStaticMarkup(
      <PageTransition id="discussion:1">
        <main>Thread</main>
      </PageTransition>,
    );
    expect(html).toMatch(/<div\b[^>]*><main>Thread<\/main><\/div>/);
  });
});

describe("PageHeader", () => {
  it("shows real state in the status slot under the title", () => {
    const html = renderToStaticMarkup(
      <PageHeader
        title="Ship the release notes"
        status={<Chip>Archived</Chip>}
      />,
    );
    expect(html).toMatch(/<h1\b[^>]*>Ship the release notes<\/h1>/);
    expect(html).toMatch(
      /<\/h1><div\b[^>]*><span\b[^>]*>Archived<\/span><\/div>/,
    );
    expect(html).not.toContain("<p");
  });

  it("renders an array of crumbs with separators and only ancestors clickable", () => {
    const html = renderToStaticMarkup(
      <PageHeader
        title="notes.md"
        crumb={[
          { label: "Library", onSelect: () => {} },
          { label: "runbooks", onSelect: () => {} },
          { label: "notes.md", onSelect: () => {} },
        ]}
      />,
    );
    expect(html).toContain('aria-label="Breadcrumb"');
    expect(html.match(/<button/g)).toHaveLength(2);
    expect(html.match(/›/g)).toHaveLength(2);
    expect(html).toContain('<span aria-current="page">notes.md</span>');
  });

  it("keeps a single crumb as a clickable back action", () => {
    const html = renderToStaticMarkup(
      <PageHeader
        title="Helper"
        crumb={{ label: "Members", onSelect: () => {} }}
      />,
    );
    expect(html).toContain("Members</button>");
    expect(html.match(/<button/g)).toHaveLength(1);
    expect(html).not.toContain('aria-current="page"');
  });

  it("exposes folder row expansion on its row action", () => {
    const html = renderToStaticMarkup(
      <RowLink primary="runbooks" expanded={false} onSelect={() => {}} />,
    );
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("runbooks");
    expect(
      renderToStaticMarkup(<RowLink primary="notes.md" onSelect={() => {}} />),
    ).not.toContain("aria-expanded");
  });

  it("renders no status slot without state", () => {
    expect(renderToStaticMarkup(<PageHeader title="Members" />)).toMatch(
      /<h1\b[^>]*>Members<\/h1><\/div>/,
    );
  });
});

describe("Table", () => {
  it("names the unlabeled actions column for assistive tech only", () => {
    const html = renderToStaticMarkup(
      <Table
        columns={[
          { key: "name", label: "Name" },
          { key: "actions", label: "" },
        ]}
        label="Members"
      >
        <tr>
          <td>You</td>
          <td />
        </tr>
      </Table>,
    );
    expect(html).toMatch(/<th[^>]*><span[^>]*>Actions<\/span><\/th>/);
  });
});
